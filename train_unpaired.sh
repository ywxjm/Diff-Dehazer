#data = '/data/lanyunwei/my_experiment_results/unpaired_cycle_gan_dehazing_dataset_new_v2_delete_bluesky'
#data = '/data/lanyunwei/my_experiment_results/unpaired_cycle_gan_dehazing_dataset_simple'
python /code/src/train_unpaired_dehaze.py \
  --output_dir="/output/" \
  --dataset_folder '/data/lanyunwei/my_experiment_results/unpaired_cycle_gan_dehazing_dataset_new_v2_delete_bluesky' \
  --train_img_prep "resize_256_randomcrop_256x256_hflip" \
  --val_img_prep "resize_512" \
  --learning_rate="1e-5" \
  --max_train_steps=100000 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=2 \
  --gradient_checkpointing \
  --lambda_gan 0.5 \
  --lambda_idt 1 \
  --lambda_cycle 1 \
  --checkpointing_steps 3000 \
  --validation_steps 3000 \
  --model_output_dir "/data/lanyunwei/my_experiment_results/img2img_turbo/model/diffdehazer_opensource"
