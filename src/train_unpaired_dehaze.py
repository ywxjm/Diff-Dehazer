import math
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import gc
import copy
import lpips
import torch
from glob import glob
import numpy as np
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPTextModel, AutoImageProcessor, AutoModelForDepthEstimation
from diffusers.optimization import get_scheduler
from peft.utils import get_peft_model_state_dict
from networks import vision_aided_loss
from model import make_1step_sched
from networks.negativeprompt.cyclegan_turbo import CycleGAN_Turbo, VAE_encode, VAE_decode, initialize_unet, initialize_vae
from my_utils.training_utils import UnpairedDataset_different_text, build_transform, parse_args_unpaired_training, UnpairedDataset_different_text_negative_prompt
from my_utils.dino_struct import DinoStructureLoss
from pathlib import Path

from BCP_CLAHE_loss import dark_channel_generate, BCCR_dehaze, fuse_block

from refine_D_network.networks import define_G
from refine_D_network import util
import datetime, swanlab
from networks.negative_prompt_text_inversion.auto_pipeline import AutoPipelineForText2Image
import torch

def main(args):

    logging_dir = Path(args.output_dir, 'logging')
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with=args.report_to,
        project_config=accelerator_project_config
    )
    set_seed(args.seed)
    swanlab.init(
        # 设置项目名
        project='diffdehazer_opensource', experiment_name=datetime.datetime.now().strftime("%Y年-%m月-%d日-%H时-%M分"))

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.model_output_dir, "checkpoints"), exist_ok=True)
    pipe = AutoPipelineForText2Image.from_pretrained("/model/lanyunwei/diffusers/sd_turbo", torch_dtype=torch.float16,
                                                     variant="fp16", use_safetensors=True)
    pipe.to("cuda")
    pipe.load_textual_inversion(
        '/code/src/inversion_embedding')
    prompt = "<haze-fog-smog-hazy-smoggy-foggy>"
    out = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0)
    negative_embedding = out[1]
    del pipe, out
    torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained("/model/lanyunwei/diffusers/sd_turbo", subfolder="tokenizer",
                                              revision=args.revision, use_fast=False, )
    noise_scheduler_1step = make_1step_sched()
    text_encoder = CLIPTextModel.from_pretrained("/model/lanyunwei/diffusers/sd_turbo", subfolder="text_encoder").cuda()
    unet, l_modules_unet_encoder, l_modules_unet_decoder, l_modules_unet_others = initialize_unet(args.lora_rank_unet,
                                                                                                  return_lora_module_names=True)
    vae_a2b, vae_lora_target_modules = initialize_vae(args.lora_rank_vae, return_lora_module_names=True)
    refine_tnet = define_G(input_nc=5, output_nc=1, ngf=32, netG='unet_trans_256', norm='instance')
    fuse_bccr_dcp = fuse_block(input_channel=3, hidden_channel=6, output_channel=3)
    weight_dtype = torch.float32
    vae_a2b.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)

    refine_tnet.to(accelerator.device, dtype=weight_dtype)
    fuse_bccr_dcp.to(accelerator.device, dtype=weight_dtype)
    text_encoder.requires_grad_(False)

    if args.gan_disc_type == "vagan_clip":
        net_disc_a = vision_aided_loss.Discriminator(cv_type='clip', loss_type=args.gan_loss_type, device="cuda")
        net_disc_a.cv_ensemble.requires_grad_(False)  # Freeze feature extractor
        net_disc_b = vision_aided_loss.Discriminator(cv_type='clip', loss_type=args.gan_loss_type, device="cuda")
        net_disc_b.cv_ensemble.requires_grad_(False)  # Freeze feature extractor

    crit_cycle, crit_idt, depth_loss = torch.nn.L1Loss(), torch.nn.L1Loss(), torch.nn.L1Loss()

    if args.enable_xformers_memory_efficient_attention:
        unet.enable_xformers_memory_efficient_attention()

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    unet.conv_in.requires_grad_(True)
    vae_b2a = copy.deepcopy(vae_a2b)
    params_gen = CycleGAN_Turbo.get_traininable_params(unet, vae_a2b, vae_b2a)

    vae_enc = VAE_encode(vae_a2b, vae_b2a=vae_b2a)
    vae_dec = VAE_decode(vae_a2b, vae_b2a=vae_b2a)

    optimizer_gen = torch.optim.AdamW(params_gen, lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2),
                                      weight_decay=args.adam_weight_decay, eps=args.adam_epsilon, )

    params_disc = list(net_disc_a.parameters()) + list(net_disc_b.parameters())
    optimizer_disc = torch.optim.AdamW(params_disc, lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2),
                                       weight_decay=args.adam_weight_decay, eps=args.adam_epsilon )
    params_other = list(fuse_bccr_dcp.parameters()) + list(refine_tnet.parameters())
    optimizer_other = torch.optim.AdamW(params_other, lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2),
                                       weight_decay=args.adam_weight_decay, eps=args.adam_epsilon )
    dataset_train = UnpairedDataset_different_text_negative_prompt(dataset_folder=args.dataset_folder, image_prep=args.train_img_prep, split="train",
                                    tokenizer=tokenizer)
    train_dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=args.train_batch_size, shuffle=True,
                                                   num_workers=args.dataloader_num_workers)
    T_val = build_transform(args.val_img_prep)

    l_images_real_test = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        l_images_real_test.extend(glob(os.path.join(args.dataset_folder, "test_real_hazy_det", 'Images', ext)))
    l_images_OHAZE_tgt = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        l_images_OHAZE_tgt.extend(glob(os.path.join(args.dataset_folder, "test_OHAZE_hazy", ext)))
    l_images_real_test, l_images_OHAZE_tgt = sorted(l_images_real_test), sorted(l_images_OHAZE_tgt)
    l_images_haze2020 = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        l_images_haze2020.extend(glob(os.path.join('/data/lanyunwei/my_experiment_results/dehaze_demo_results/hazy/hazy2020', ext)))

    # A 干净图像 B 有雾图像

    lr_scheduler_gen = get_scheduler(args.lr_scheduler, optimizer=optimizer_gen,
                                     num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
                                     num_training_steps=args.max_train_steps * accelerator.num_processes,
                                     num_cycles=args.lr_num_cycles, power=args.lr_power)
    lr_scheduler_disc = get_scheduler(args.lr_scheduler, optimizer=optimizer_disc,
                                      num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
                                      num_training_steps=args.max_train_steps * accelerator.num_processes,
                                      num_cycles=args.lr_num_cycles, power=args.lr_power)
    lr_schedule_other = get_scheduler(args.lr_scheduler, optimizer=optimizer_other,
                                      num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
                                      num_training_steps=args.max_train_steps * accelerator.num_processes,
                                      num_cycles=args.lr_num_cycles, power=args.lr_power)

    net_lpips = lpips.LPIPS(net='vgg')
    net_lpips.cuda()
    net_lpips.requires_grad_(False)

    unet, vae_enc, vae_dec, net_disc_a, net_disc_b, fuse_bccr_dcp, refine_tnet  = accelerator.prepare(
        unet, vae_enc, vae_dec, net_disc_a, net_disc_b, fuse_bccr_dcp, refine_tnet)

    net_lpips, optimizer_gen, optimizer_disc, optimizer_other, train_dataloader, lr_scheduler_gen, lr_scheduler_disc, lr_schedule_other = accelerator.prepare(
        net_lpips, optimizer_gen, optimizer_disc, optimizer_other, train_dataloader, lr_scheduler_gen,
        lr_scheduler_disc, lr_schedule_other)

    first_epoch = 0
    global_step = 0
    progress_bar = tqdm(range(0, args.max_train_steps), initial=global_step, desc="Steps",
                        disable=not accelerator.is_local_main_process, )
    # turn off eff. attn for the disc
    for name, module in net_disc_a.named_modules():
        if "attn" in name:
            module.fused_attn = False
    for name, module in net_disc_b.named_modules():
        if "attn" in name:
            module.fused_attn = False

    for epoch in range(first_epoch, args.max_train_epochs):
        for step, batch in enumerate(train_dataloader):
            l_acc = [unet, net_disc_a, net_disc_b, vae_enc, vae_dec]
            with accelerator.accumulate(*l_acc):
                fixed_a2b_tokens = batch["input_ids_src"][0]
                fixed_a2b_emb_base = text_encoder(fixed_a2b_tokens.cuda().unsqueeze(0))[0].detach()
                fixed_b2a_tokens = batch["input_ids_tgt"][0]
                negative_token = batch["negative_ids"][0]
                negative_token_a2b = batch["negative_ids_a2b"][0]
                fixed_b2a_emb_base = text_encoder(fixed_b2a_tokens.cuda().unsqueeze(0))[0].detach()
                # negative_emb_base = text_encoder(negative_token.cuda().unsqueeze(0))[0].detach() #haze生成的embedding
                negative_emb_base = negative_embedding # textual inversion 的embedding
                negative_emb_base_a2b = text_encoder(negative_token_a2b.cuda().unsqueeze(0))[0].detach()
                prompt_emb_b2a = torch.cat([negative_emb_base, fixed_b2a_emb_base])
                prompt_emb_a2b = torch.cat([negative_emb_base_a2b, fixed_a2b_emb_base])

                # free up some memory
                img_a = batch["pixel_values_src"].to(dtype=weight_dtype)  # clear
                img_b = batch["pixel_values_tgt"].to(dtype=weight_dtype)  # haze
                img_b_0_1 = (img_b + 1) / 2.
                J_dcp_b, t_dcp_b, A_dcp_b = dark_channel_generate(img_b_0_1)
                J_dcp_b = torch.clamp(J_dcp_b, 0, 1)
                J_dcp_normalize = (J_dcp_b - 0.5) / 0.5

                J_BCCR_b, T_BCCR_b, J_BCCR_tensor_b, T_BCCR_tensor_b = BCCR_dehaze(img_b_0_1)
                J_BCCR_tensor_b = torch.clamp(J_BCCR_tensor_b, 0, 1).to('cuda').to(dtype=weight_dtype)
                T_BCCR_tensor_b = T_BCCR_tensor_b.to('cuda').to(dtype=weight_dtype)
                J_BCCR_tensor_normalize = (J_BCCR_tensor_b - 0.5) / 0.5

                fuse_output = fuse_bccr_dcp(J_dcp_normalize, J_BCCR_tensor_normalize, img_b)
                bsz = img_a.shape[0]

                timesteps = torch.tensor([noise_scheduler_1step.config.num_train_timesteps - 1] * bsz,
                                         device=img_a.device).long()
                """
                Cycle Objective
                """

                cyc_fake_b = CycleGAN_Turbo.forward_with_networks(img_a, "a2b", vae_enc, unet, vae_dec,
                                                                  noise_scheduler_1step, timesteps, prompt_emb_a2b)

                cyc_rec_a = CycleGAN_Turbo.forward_with_networks(cyc_fake_b, "b2a", vae_enc, unet, vae_dec,
                                                                 noise_scheduler_1step, timesteps, prompt_emb_b2a)

                loss_cycle_a = crit_cycle(cyc_rec_a, img_a) * args.lambda_cycle
                loss_cycle_a += net_lpips(cyc_rec_a, img_a).mean() * args.lambda_cycle_lpips
                # B -> fake A -> rec B
                cyc_fake_a = CycleGAN_Turbo.forward_with_networks(img_b, "b2a", vae_enc, unet, vae_dec,
                                                                  noise_scheduler_1step, timesteps, prompt_emb_b2a)
                cyc_rec_b = CycleGAN_Turbo.forward_with_networks(cyc_fake_a, "a2b", vae_enc, unet, vae_dec,
                                                                 noise_scheduler_1step, timesteps, prompt_emb_a2b)
                refine_out = CycleGAN_Turbo.forward_with_networks(fuse_output, "b2a", vae_enc, unet, vae_dec,
                                                                  noise_scheduler_1step, timesteps, prompt_emb_b2a)

                refine_T, out_T = refine_tnet(torch.cat((img_b, t_dcp_b, T_BCCR_tensor_b), dim=1))


                shape = refine_out.shape
                dcp_A_scale = A_dcp_b
                map_A = (dcp_A_scale).reshape((1, 3, 1, 1)).repeat(1, 1, shape[2], shape[3])

                refine_T_map = refine_T.repeat(1, 3, 1, 1)
                rec_dcp_b = util.synthesize_fog(refine_out, refine_T_map, map_A)

                ### optional
                cyc_fake_b_0_1 = (cyc_fake_b + 1) / 2.
                cyc_fake_b_0_1 = cyc_fake_b_0_1.detach()
                J_BCCR_fake_b, T_BCCR_fake_b, J_BCCR_tensor_fake_b, T_BCCR_tensor_fake_b = BCCR_dehaze(cyc_fake_b_0_1)
                J_BCCR_tensor_fake_b = torch.clamp(J_BCCR_tensor_fake_b, 0, 1).to('cuda').to(dtype=weight_dtype)
                T_BCCR_tensor_fake_b = T_BCCR_tensor_fake_b.to('cuda').to(dtype=weight_dtype)
                J_BCCR_tensor_fake_b_normalize = (J_BCCR_tensor_fake_b - 0.5) / 0.5
                J_dcp_fake_b, t_dcp_fake_b, A_dcp_fake_b = dark_channel_generate(cyc_fake_b_0_1)
                J_dcp_fake_b = torch.clamp(J_dcp_fake_b, 0, 1)
                J_dcp_fake_b_normalize = (J_dcp_fake_b -0.5) / 0.5
                fuse_output_fake_b = fuse_bccr_dcp(J_dcp_fake_b_normalize, J_BCCR_tensor_fake_b_normalize, cyc_fake_b.detach())
                refine_T_fake_b, out_T_fake_b = refine_tnet(torch.cat((cyc_fake_b.detach(), t_dcp_fake_b, T_BCCR_tensor_fake_b), dim=1))
                torch.cuda.empty_cache()

                refine_out_fake_b = CycleGAN_Turbo.forward_with_networks(fuse_output_fake_b, "b2a", vae_enc, unet, vae_dec,
                                                                  noise_scheduler_1step, timesteps, prompt_emb_b2a)
                fake_b_A_scale = A_dcp_fake_b
                map_A_fake_b = (fake_b_A_scale).reshape((1, 3, 1, 1)).repeat(1, 1, shape[2], shape[3])
                refine_T_fake_map = refine_T_fake_b.repeat(1, 3, 1, 1)
                rec_dcp_fake_b = util.synthesize_fog(refine_out_fake_b, refine_T_fake_map, map_A_fake_b)

                loss_dcp_fake_b = crit_cycle(rec_dcp_fake_b, cyc_fake_b) * args.lambda_cycle
                loss_dcp_fake_b += net_lpips(rec_dcp_fake_b, cyc_fake_b).mean() * args.lambda_cycle_lpips

                ### optional stop

                loss_dcp_b = crit_cycle(rec_dcp_b, img_b) * args.lambda_cycle
                loss_dcp_b += net_lpips(rec_dcp_b, img_b).mean() * args.lambda_cycle_lpips



                loss_cycle_b = crit_cycle(cyc_rec_b, img_b) * args.lambda_cycle
                loss_cycle_b += net_lpips(cyc_rec_b, img_b).mean() * args.lambda_cycle_lpips
                accelerator.backward(loss_cycle_a + loss_cycle_b + 0.5 * loss_dcp_b + 0.1 * loss_dcp_fake_b,
                                     retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_gen, args.max_grad_norm)
                    accelerator.clip_grad_norm_(params_other, args.max_grad_norm)

                optimizer_gen.step()
                lr_scheduler_gen.step()
                optimizer_gen.zero_grad()

                optimizer_other.step()
                lr_schedule_other.step()
                optimizer_other.zero_grad()

                """
                Generator Objective (GAN) for task a->b and b->a (fake inputs)
                """
                fake_a = CycleGAN_Turbo.forward_with_networks(img_b, "b2a", vae_enc, unet, vae_dec,
                                                              noise_scheduler_1step, timesteps, prompt_emb_b2a)

                fake_b = CycleGAN_Turbo.forward_with_networks(img_a, "a2b", vae_enc, unet, vae_dec,
                                                              noise_scheduler_1step, timesteps, prompt_emb_a2b)
                loss_gan_a = net_disc_a(fake_a, for_G=True).mean() * args.lambda_gan

                loss_gan_b = net_disc_b(fake_b, for_G=True).mean() * args.lambda_gan
                accelerator.backward(loss_gan_a + loss_gan_b, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_gen, args.max_grad_norm)
                optimizer_gen.step()
                lr_scheduler_gen.step()
                optimizer_gen.zero_grad()


                """
                Identity Objective
                """
                # a2b haze b2a, clear
                idt_a = CycleGAN_Turbo.forward_with_networks(img_b, "a2b", vae_enc, unet, vae_dec,
                                                             noise_scheduler_1step, timesteps, prompt_emb_a2b)
                loss_idt_a = crit_idt(idt_a, img_b) * args.lambda_idt
                loss_idt_a += net_lpips(idt_a, img_b).mean() * args.lambda_idt_lpips
                idt_b = CycleGAN_Turbo.forward_with_networks(img_a, "b2a", vae_enc, unet, vae_dec,
                                                             noise_scheduler_1step, timesteps, prompt_emb_b2a)
                loss_idt_b = crit_idt(idt_b, img_a) * args.lambda_idt
                loss_idt_b += net_lpips(idt_b, img_a).mean() * args.lambda_idt_lpips
                loss_g_idt = loss_idt_a + loss_idt_b
                accelerator.backward(loss_g_idt, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_gen, args.max_grad_norm)
                optimizer_gen.step()
                lr_scheduler_gen.step()
                optimizer_gen.zero_grad()

                """
                Discriminator for task a->b and b->a (fake inputs)
                """
                loss_D_A_fake = net_disc_a(fake_a.detach(), for_real=False).mean() * args.lambda_gan
                loss_D_B_fake = net_disc_b(fake_b.detach(), for_real=False).mean() * args.lambda_gan

                loss_D_fake = (loss_D_A_fake + loss_D_B_fake) * 0.5
                accelerator.backward(loss_D_fake, retain_graph=False)
                if accelerator.sync_gradients:
                    params_to_clip = list(net_disc_a.parameters()) + list(net_disc_b.parameters())
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer_disc.step()
                lr_scheduler_disc.step()
                optimizer_disc.zero_grad()

                """
                Discriminator for task a->b and b->a (real inputs)
                """
                loss_D_A_real = net_disc_a(img_a, for_real=True).mean() * args.lambda_gan
                loss_D_B_real = net_disc_b(img_b, for_real=True).mean() * args.lambda_gan
                loss_D_real = (loss_D_A_real + loss_D_B_real) * 0.5
                accelerator.backward(loss_D_real, retain_graph=False)
                if accelerator.sync_gradients:
                    params_to_clip = list(net_disc_a.parameters()) + list(net_disc_b.parameters())
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer_disc.step()
                lr_scheduler_disc.step()
                optimizer_disc.zero_grad()


            logs = {}
            logs["cycle_a"] = loss_cycle_a.detach().item()
            logs["cycle_b"] = loss_cycle_b.detach().item()
            logs["gan_a"] = loss_gan_a.detach().item()
            logs["gan_b"] = loss_gan_b.detach().item()
            logs["disc_a"] = loss_D_A_fake.detach().item() + loss_D_A_real.detach().item()
            logs["disc_b"] = loss_D_B_fake.detach().item() + loss_D_B_real.detach().item()
            logs["idt_a"] = loss_idt_a.detach().item()
            logs["idt_b"] = loss_idt_b.detach().item()

            swanlab.log(logs)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    eval_unet = accelerator.unwrap_model(unet)
                    eval_vae_enc = accelerator.unwrap_model(vae_enc)
                    eval_vae_dec = accelerator.unwrap_model(vae_dec)
                    eval_fuse_net = accelerator.unwrap_model(fuse_bccr_dcp)
                    eval_refine_tnet = accelerator.unwrap_model(refine_tnet)


                    if global_step % args.checkpointing_steps == 1:
                        outf = os.path.join(args.model_output_dir, "checkpoints", f"model_{epoch}_{global_step}.pkl")
                        sd = {}
                        sd["l_target_modules_encoder"] = l_modules_unet_encoder
                        sd["l_target_modules_decoder"] = l_modules_unet_decoder
                        sd["l_modules_others"] = l_modules_unet_others
                        sd["rank_unet"] = args.lora_rank_unet
                        sd["sd_encoder"] = get_peft_model_state_dict(eval_unet, adapter_name="default_encoder")
                        sd["sd_decoder"] = get_peft_model_state_dict(eval_unet, adapter_name="default_decoder")
                        sd["sd_other"] = get_peft_model_state_dict(eval_unet, adapter_name="default_others")
                        sd["rank_vae"] = args.lora_rank_vae
                        sd["vae_lora_target_modules"] = vae_lora_target_modules
                        sd["sd_vae_enc"] = eval_vae_enc.state_dict()
                        sd["sd_vae_dec"] = eval_vae_dec.state_dict()
                        sd["fuse_block"] = eval_fuse_net.state_dict()
                        sd["refine_tnet"] = eval_refine_tnet.state_dict()
                        torch.save(sd, outf)
                        gc.collect()
                        torch.cuda.empty_cache()



            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            if global_step >= args.max_train_steps:
                break


if __name__ == "__main__":
    args = parse_args_unpaired_training()
    main(args)
