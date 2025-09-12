import glob
import os
import argparse
from PIL import Image
import torch
from torchvision import transforms
from tqdm.auto import tqdm
from networks.negativeprompt.cyclegan_turbo_test import CycleGAN_Turbo
from my_utils.training_utils import build_transform
from transformers import AutoTokenizer, CLIPTextModel
from tqdm import tqdm
from networks.negative_prompt_text_inversion.auto_pipeline import AutoPipelineForText2Image
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', type=str, required=False, help='path to the input image')
    parser.add_argument('--prompt', type=str, required=False, help='the prompt to be used. It is required when loading a custom model_path.')
    parser.add_argument('--model_name', type=str, default=None, help='name of the pretrained model to be used')
    parser.add_argument('--model_path', type=str, default=None, help='path to a local model state dict to be used')
    parser.add_argument('--output_dir', type=str, default='/output/', help='the directory to save the output')
    parser.add_argument('--image_prep', type=str, default='resize_512', help='the image preparation method')
    parser.add_argument('--input_folder', type=str, default=None, help='the directory to input_folder_clear')
    parser.add_argument('--input_folder_haze', type=str, default=None, help='the directory to input_folder_haze')
    parser.add_argument('--checkpoint_dir', type=str, default=None, help='the image preparation method')
    parser.add_argument('--checkpoint_name', type=str, default=None, help='the image preparation method')
    parser.add_argument('--direction', type=str, default=None, help='the direction of translation. None for pretrained models, a2b or b2a for custom paths.')
    parser.add_argument('--caption_name', type=str, default=None, help='the image preparation method')
    parser.add_argument("--revision", default=None, type=str)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained("/model/lanyunwei/diffusers/sd_turbo", subfolder="tokenizer",
                                              revision=args.revision, use_fast=False, )
    text_encoder = CLIPTextModel.from_pretrained("/model/lanyunwei/diffusers/sd_turbo", subfolder="text_encoder").cuda()

    text_encoder.requires_grad_(False)
    # only one of model_name and model_path should be provided
    if args.model_name is None != args.model_path is None:
        raise ValueError('Either model_name or model_path should be provided')

    if args.model_path is not None and args.prompt is None:
        raise ValueError('prompt is required when loading a custom model_path.')

    if args.model_name is not None:
        assert args.prompt is None, 'prompt is not required when loading a pretrained model.'
        assert args.direction is None, 'direction is not required when loading a pretrained model.'
    import torch

    if torch.cuda.is_available():
        print("A GPU is available.")
    else:
        print("No GPU is available")

    # initialize the model
    outf = os.path.join(args.checkpoint_dir, args.checkpoint_name)
    model = CycleGAN_Turbo(pretrained_name=args.model_name, pretrained_path=args.model_path, checkpoint_name=args.checkpoint_name, checkpoint_dir=args.checkpoint_dir)
    model.eval()
    # model.unet.enable_xformers_memory_efficient_attention()
    weight_dtype = torch.float32
    T_val = build_transform(args.image_prep)

    ### inversion

    pipe = AutoPipelineForText2Image.from_pretrained("/model/lanyunwei/diffusers/sd_turbo", torch_dtype=torch.float16,
                                                     variant="fp16", use_safetensors=True)
    pipe.to("cuda")
    pipe.load_textual_inversion(
        '/code/src/inversion_embedding')
    prompt = "<haze-fog-smog-hazy-smoggy-foggy>"
    out = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0)
    negative_embedding = out[1]

    del pipe, out

    test_real_hazy_text = os.path.join('/data/lanyunwei/my_experiment_results/caption_list_dehaze',
                                       args.caption_name)

    with open(test_real_hazy_text, "r") as f:
        test_real_hazy_lines = f.readlines()

    list = []
    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    for extension in image_extensions:
        # 使用glob模块获取所有匹配的图片文件
        list += glob.glob(os.path.join(args.input_folder, extension))

    for img in tqdm(list):
        img_name = img.split('/')[-1].split('.')[0]
        for line in test_real_hazy_lines:
            line = line.strip()
            test_name = line.split('!')[0].split('.')[0]
            test_caption = line.split('!')[1]
            if test_name == img_name:
                image_caption = test_caption
                break
        fixed_b2a_tokens = \
            tokenizer(image_caption, max_length=tokenizer.model_max_length, padding="max_length", truncation=True,
                      return_tensors="pt").input_ids[0]
        fixed_b2a_emb_base = text_encoder(fixed_b2a_tokens.cuda().unsqueeze(0))[0].detach()
        negative_emb_base = negative_embedding
        prompt_emb_base = torch.cat([negative_emb_base, fixed_b2a_emb_base])

        input_image = Image.open(img).convert('RGB')
        if input_image.height >= 1000 or input_image.width >= 1000:
            input_image = input_image.resize((512, 512), Image.LANCZOS)
        new_h = (input_image.height // 8) * 8
        new_w = (input_image.width // 8) * 8
        resized_img = input_image.resize((new_w, new_h), Image.LANCZOS)
        # translate the image
        with torch.no_grad():

            input_img = T_val(resized_img)
            img_b = transforms.ToTensor()(input_img)
            img_b = transforms.Normalize([0.5], [0.5])(img_b).unsqueeze(0).cuda()

            output = model(img_b, direction=args.direction, caption_emb=prompt_emb_base)



        output_pil = transforms.ToPILImage()(output[0].cpu() * 0.5 + 0.5)
        output_pil = output_pil.resize((input_image.width, input_image.height), Image.LANCZOS)

        # save the output image
        bname = os.path.basename(img)
        os.makedirs(args.output_dir, exist_ok=True)
        output_pil.save(os.path.join(args.output_dir, bname))







