from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch, argparse
import glob, os
from tqdm import tqdm
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default='/code/haze_512×512/')
parser.add_argument('--text_pth', type=str, default='/data/lanyunwei/my_experiment_results/capation_list/caption_test.txt')
args = parser.parse_args()

def write_lists_to_file(text_pth, list1, list2):
    with open(text_pth, 'w') as f:
        for item1, item2 in zip(list1, list2):
            f.write(str(item1) + '!' + str(item2))
            f.write("\r\n")


device = "cuda" if torch.cuda.is_available() else "cpu"

processor = Blip2Processor.from_pretrained("/data/lanyunwei/my_experiment_results/model_publish/BLIP2/")
model = Blip2ForConditionalGeneration.from_pretrained("/data/lanyunwei/my_experiment_results/model_publish/BLIP2/", torch_dtype=torch.float16, use_safetensors=False)
model.to(device)





list = []
caption = []
img_pth = []
image_extensions = ['*.jpg', '*.jpeg', '*.png']
for extension in image_extensions:
    # 使用glob模块获取所有匹配的图片文件
    list += glob.glob(os.path.join(args.input_dir, extension))

for img in tqdm(list):
    image = Image.open(img).convert('RGB')
    image_name = img.split('/')[-1]
    inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(**inputs)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    caption.append(generated_text)
    img_pth.append(image_name)

write_lists_to_file(args.text_pth, img_pth, caption)



