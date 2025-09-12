import os

import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage, Normalize, Compose
from BCCR_dehaze import image_dehazer
class GuidedFilter(nn.Module):
    def __init__(self, r=40, eps=1e-3, gpu_ids=None):  # only work for gpu case at this moment
        super(GuidedFilter, self).__init__()
        self.r = r
        self.eps = eps
        # self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU

        self.boxfilter = nn.AvgPool2d(kernel_size=2 * self.r + 1, stride=1, padding=self.r)

    def forward(self, I, p):
        """
        I -- guidance image, should be [0, 1]
        p -- filtering input image, should be [0, 1]
        """

        # N = self.boxfilter(self.tensor(p.size()).fill_(1))
        N = self.boxfilter(torch.ones(p.size()))

        if I.is_cuda:
            N = N.cuda()

        # print(N.shape)
        # print(I.shape)
        # print('-----------')

        mean_I = self.boxfilter(I) / N
        mean_p = self.boxfilter(p) / N
        mean_Ip = self.boxfilter(I * p) / N
        cov_Ip = mean_Ip - mean_I * mean_p

        mean_II = self.boxfilter(I * I) / N
        var_I = mean_II - mean_I * mean_I

        a = cov_Ip / (var_I + self.eps)
        b = mean_p - a * mean_I
        mean_a = self.boxfilter(a) / N
        mean_b = self.boxfilter(b) / N

        return mean_a * I + mean_b

def get_dark_channel(input, win_size=5, ):
    _, _, H, W = input.shape
    _, _, H, W = input.shape
    maxpool = nn.MaxPool3d((3, win_size, win_size), stride=1, padding=(0, win_size // 2, win_size // 2))
    dc = maxpool(0 - input[:, :, :, :])

    return -dc

def get_bright_channel(I, w=5):
    _, _, H, W = I.shape
    maxpool = nn.MaxPool3d((3, w, w), stride=1, padding=(0, w // 2, w // 2))
    bc = maxpool(I[:, :, :, :])

    return bc


def atmospheric_light(self, img, dark_img):
    num, chl, height, width = img.shape
    topNum = int(0.01 * height * width)

    A = torch.Tensor(num, chl, 1, 1)
    if img.is_cuda:
        A = A.cuda()

    for num_id in range(num):
        curImg = img[num_id, ...]
        curDarkImg = dark_img[num_id, 0, ...]

        _, indices = curDarkImg.reshape([height * width]).sort(descending=True)
        # curMask = indices < topNum

        for chl_id in range(chl):
            imgSlice = curImg[chl_id, ...].reshape([height * width])
            A[num_id, chl_id, 0, 0] = torch.mean(imgSlice[indices[0:topNum]])

    return A


def get_atmosphere_dark(I, dark_ch, p=0.01):
    B, _, H, W = dark_ch.shape
    num_pixel = int(p * H * W)
    flat_dc = dark_ch.resize(B, H * W)
    flat_I = I.resize(B, 3, H * W)
    index = torch.argsort(flat_dc, descending=True)[:, :num_pixel]
    A = torch.zeros((B, 3)).to('cuda')

    for i in range(B):
        A[i] = flat_I[i, :, index].mean((1, 2))

    return A


def get_atmosphere_bright(I, bright_ch, p=0.01):
    B, _, H, W = bright_ch.shape
    num_pixel = int(p * H * W)
    flat_bc = bright_ch.resize(B, H * W)
    flat_I = I.resize(B, 3, H * W)
    index = torch.argsort(flat_bc, descending=False)[:, :num_pixel]
    A = torch.zeros((B, 3)).to('cuda')

    for i in range(B):
        A[i] = flat_I[i, :, index].mean((1, 2))

    return A


def bright_channel_generate(img, w=5, p=0.01):
    bright_channel = get_bright_channel(img, w)

    A = get_atmosphere_bright(img, bright_channel, p)

    norm_I = (1 - img) / (1 - A[:, :, None, None] + 1e-6)
    bright_channel = get_bright_channel(norm_I, w)
    t_slide = (1 - 0.95 * bright_channel)

    # t1 = t_slide[dc < 0.6]
    # t2 = T[dc < 0.6]
    # if len(t1) == 0:
    #    return 0
    # print(T.shape, t1.shape)


    return t_slide

def dark_channel_generate(x, omega=0.95):
    guided_filter = GuidedFilter(r=15, eps=1e-3)
    if x.shape[1] > 1:
            # rgb2gray
        guidance = 0.2989 * x[:,0,:,:] + 0.5870 * x[:,1,:,:] + 0.1140 * x[:,2,:,:]
    else:
        guidance = x
        # rescale to [0,1]
    guidance = (guidance + 1) / 2
    guidance = torch.unsqueeze(guidance, dim=1)
    imgPatch = (x + 1) / 2

    num, chl, height, width = imgPatch.shape

    # dark_img and A with the range of [0,1]
    dark_img = get_dark_channel(imgPatch)
    A = get_atmosphere_dark(imgPatch, dark_img)
    A = A.unsqueeze(-1).unsqueeze(-1)
    map_A = A.repeat(1, 1, height, width)
    # make sure channel of trans_raw == 1
    trans_raw = 1 - omega * get_dark_channel(imgPatch / map_A)

    # get initial results
    T_DCP = guided_filter(guidance, trans_raw)
    J_DCP = (imgPatch - map_A) / T_DCP.repeat(1, 3, 1, 1) + map_A

    return J_DCP, T_DCP, torch.squeeze(A)

def BCCR_dehaze(x):
    BCCR_img_dehazer = image_dehazer()
    t2PIL = Compose([
        ToPILImage()
    ])
    x_pil = t2PIL(x[0])
    J_BCCR, T_BCCR = BCCR_img_dehazer.remove_haze(x_pil)
    return J_BCCR, T_BCCR

if __name__ == '__main__':
    import glob
    from tqdm.auto import tqdm
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=False, help='path to the input image')
    parser.add_argument('--output_dir', type=str, required=False,
                        help='the prompt to be used. It is required when loading a custom model_path.')
    parser.add_argument('--model_dcp', type=bool, default=False, required=False,
                        help='the prompt to be used. It is required when loading a custom model_path.')
    args = parser.parse_args()
    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.output_dir, exist_ok=True)
    list = []
    for extension in image_extensions:
        # 使用glob模块获取所有匹配的图片文件
        list += glob.glob(os.path.join(args.input_dir, extension))
    for img_pth in tqdm(list):
        img_name = img_pth.split('/')[-1]
        img = Image.open(img_pth).convert('RGB')
        if img.height >= 1000 or img.width >= 1000:
            img = img.resize((1024, 1024), Image.LANCZOS)
        input_transfer = Compose([
            ToTensor()
        ])
        img = input_transfer(img)
        img = img.unsqueeze(0).to(device)
        save_dir = os.path.join(args.output_dir, img_name)
        if args.model_dcp:
            J_DCP, T_DCP, A = dark_channel_generate(img)
            J_DCP = torch.clamp(J_DCP, 0, 1)
            J_DCP_PIL = ToPILImage()(J_DCP[0])
            J_DCP_PIL.save(save_dir)
        else:
            BCCR_img_dehazer = image_dehazer()
            images = img.cpu().permute(0, 2, 3, 1).float().numpy()
            images = (images * 255).round().astype("uint8")

            J_BCCR, T_BCCR = BCCR_img_dehazer.remove_haze(images[0])
            J_BCCR_tensor = (J_BCCR / 255.0)
            J_BCCR_tensor = torch.from_numpy(J_BCCR_tensor.transpose(2, 0, 1)).unsqueeze(0)
            T_BCCR_tensor = torch.from_numpy(T_BCCR).unsqueeze(0).unsqueeze(0)
            J_BCCR_tensor = torch.clamp(J_BCCR_tensor, 0, 1)
            J_BCCR_pil = ToPILImage()(J_BCCR_tensor[0])
            J_BCCR_pil.save(save_dir)





    # image_dehazer = image_dehazer()
    # img_pth = '/data/lanyunwei/my_experiment_results/unpsired_cycle_gan_dehazing_dataset_new_v3/test_SOTS_outdoor_hazy/0001.png'
    #
    # output_dir = '/output/'
    # save_dir = os.path.join(output_dir, img_name)
    # img = Image.open(img_pth).convert('RGB')
    #
    # output_transfer = Compose([
    #     ToPILImage()
    # ])
    # J_BCCR, T_BCCR = BCCR_dehaze(img)
    # print(J_BCCR)

    # dark_channel = get_bright_channel(img)
    # bright_channel = get_bright_channel(img)
    # A_dark = get_atmosphere_dark(img, dark_channel)
    # A_bright = get_atmosphere_bright(img, bright_channel)
    # J_DCP, T_DCP, A = dark_channel_generate(img)
    # J_DCP = torch.clamp(T_DCP, 0, 1)
    # J_DCP = output_transfer(J_DCP[0])
    # J_DCP.save(save_dir)
    # t_slide = bright_channel_generate(img)
    # t_slide = output_transfer(t_slide[0])
    # t_slide.save(save_dir)



