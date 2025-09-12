
python /code/src/inference_unpaired_dehaze.py \
--model_name "hazy_to_clear" \
--input_folder "/data/hazy2020/" \
--output_dir "/output/inversion_haze/21001/" \
--image_prep "no_resize" \
--caption_name "/code/assets/text_prompt_new/capation_2020.txt" \
--checkpoint_name "model_net.pkl" \
--checkpoint_dir "/data/lanyunwei/my_experiment_results/img2img_turbo/model/diffdehazer_opensource/checkpoints/"

python /code/src/inference_unpaired_dehaze.py \
--model_name "hazy_to_clear" \
--input_folder "/data/urhi/" \
--output_dir "/output/inversion_haze/21001/" \
--image_prep "no_resize" \
--caption_name "/code/assets/text_prompt_new/capation_urhi.txt" \
--checkpoint_name "model_net.pkl" \
--checkpoint_dir "/data/lanyunwei/my_experiment_results/img2img_turbo/model/diffdehazer_opensource/checkpoints/"

python /code/src/inference_unpaired_dehaze.py \
--model_name "hazy_to_clear" \
--input_folder "/data/rtts/" \
--output_dir "/output/inversion_haze/21001/" \
--image_prep "no_resize" \
--caption_name "/code/assets/text_prompt_new/capation_rtts.txt" \
--checkpoint_name "model_net.pkl" \
--checkpoint_dir "/data/lanyunwei/my_experiment_results/img2img_turbo/model/diffdehazer_opensource/checkpoints/"

