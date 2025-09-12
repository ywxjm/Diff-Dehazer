# Diff-Dehazer (AAAI 2025)
Exploiting Diffusion Prior for Real-World Image Dehazing with Unpaired Training

## Abstract
Unpaired training has been verified as one of the most effective paradigms for real scene dehazing by learning from unpaired real-world hazy and clear images. Although numerous studies have been proposed, current methods demonstrate limited generalization for various real scenes due to limited feature representation and insufficient use of real-world prior.
Inspired by the strong generative capabilities of diffusion models in producing both hazy and clear images, we exploit diffusion prior for real-world image dehazing, and propose an unpaired framework named Diff-Dehazer. Specifically, we leverage diffusion prior as bijective mapping learners within the CycleGAN, a classic unpaired learning framework.
Considering that physical priors contain pivotal statistics information of real-world data, we further excavate real-world knowledge by integrating physical priors into our framework.
Furthermore, we introduce a new perspective for adequately leveraging the representation ability of diffusion models by removing degradation in image and text modalities, so as to improve the dehazing effect.
Extensive experiments on multiple real-world datasets demonstrate the superior performance of our method.

## Requirements
python 3.10 torch 2.1
bash environment.sh
you need to download sdturbov2.1 from https://huggingface.co/stabilityai/sd-turbo/tree/main and change its path


## Download training Dataset 

Our training dataset can be downloaded in train_data.txt

## Pretrained weight 

The pre-trained weights can be found in the /pretrained/.pretrained.txt.  

## Training / testing on Dataset
Start the training process by running the following command:
```sh
bash /train_unpaired.sh 
```
you need to change the path of --output_dir, --dataset_folder, --model_output_dir
Run the testing script:
```sh
bash /test_unpaired.sh
```
you need to change the path of --input_folder, --output_dir, --checkpoint_name, --checkpoint_dir
