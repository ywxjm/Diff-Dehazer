import os

import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage, Normalize, Compose
from networks.BCCR_dehaze import image_dehazer

import cv2
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


# def get_bright_channel(I, w=5):
#     _, _, H, W = I.shape
#     maxpool = nn.MaxPool3d((3, w, w), stride=1, padding=(0, w // 2, w // 2))
#     bc = maxpool(I[:, :, :, :])
#
#     return bc




def get_atmosphere_dark(I, dark_ch, p=0.01):
    B, _, H, W = dark_ch.shape
    num_pixel = int(p * H * W)
    flat_dc = dark_ch.resize(B, H * W)
    flat_I = I.resize(B, 3, H * W)
    index = torch.argsort(flat_dc, descending=True)[:, :num_pixel]
    A = torch.zeros((B, 3)).to('cuda')

    for i in range(B):
        A[i] = flat_I[i, :, index[i][torch.argsort(torch.max(flat_I[i][:, index[i]], 0)[0], descending=True)[0]]]

    return A


# def get_atmosphere2(I, bright_ch, p=0.0001):
#     B, _, H, W = bright_ch.shape
#     num_pixel = int(p * H * W)
#     flat_bc = bright_ch.resize(B, H * W)
#     flat_I = I.resize(B, 3, H * W)
#     index = torch.argsort(flat_bc, descending=False)[:, :num_pixel]
#     A = torch.zeros((B, 3)).to('cuda')
#
#     for i in range(B):
#         A[i] = flat_I[i, :, index].mean((1, 2))
#
#     return A

def dark_channel_generate(x, omega=0.95):
    guided_filter = GuidedFilter(r=15, eps=1e-3)
    if x.shape[1] > 1:
            # rgb2gray
        guidance = 0.2989 * x[:,0,:,:] + 0.5870 * x[:,1,:,:] + 0.1140 * x[:,2,:,:]
    else:
        guidance = x

    # guidance = (guidance + 1) / 2
    guidance = torch.unsqueeze(guidance, dim=1)
    # imgPatch = (x + 1) / 2
    imgPatch = x

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

def fuse_images(real_I, rec_J, refine_J):
    """
    real_I, rec_J, and refine_J: Images with shape hxwx3
    """
    # realness features
    mat_RGB2YMN = np.array([[0.299,0.587,0.114],
                            [0.30,0.04,-0.35],
                            [0.34,-0.6,0.17]])

    recH,recW,recChl = rec_J.shape
    rec_J_flat = rec_J.reshape([recH*recW,recChl])
    rec_J_flat_YMN = (mat_RGB2YMN.dot(rec_J_flat.T)).T
    rec_J_YMN = rec_J_flat_YMN.reshape(rec_J.shape)

    refine_J_flat = refine_J.reshape([recH*recW,recChl])
    refine_J_flat_YMN = (mat_RGB2YMN.dot(refine_J_flat.T)).T
    refine_J_YMN = refine_J_flat_YMN.reshape(refine_J.shape)

    real_I_flat = real_I.reshape([recH*recW,recChl])
    real_I_flat_YMN = (mat_RGB2YMN.dot(real_I_flat.T)).T
    real_I_YMN = real_I_flat_YMN.reshape(real_I.shape)

    # gradient features
    rec_Gx = cv2.Sobel(rec_J_YMN[:,:,0],cv2.CV_64F,1,0,ksize=3)
    rec_Gy = cv2.Sobel(rec_J_YMN[:,:,0],cv2.CV_64F,0,1,ksize=3)
    rec_GM = np.sqrt(rec_Gx**2 + rec_Gy**2)

    refine_Gx = cv2.Sobel(refine_J_YMN[:,:,0],cv2.CV_64F,1,0,ksize=3)
    refine_Gy = cv2.Sobel(refine_J_YMN[:,:,0],cv2.CV_64F,0,1,ksize=3)
    refine_GM = np.sqrt(refine_Gx**2 + refine_Gy**2)

    real_Gx = cv2.Sobel(real_I_YMN[:,:,0],cv2.CV_64F,1,0,ksize=3)
    real_Gy = cv2.Sobel(real_I_YMN[:,:,0],cv2.CV_64F,0,1,ksize=3)
    real_GM = np.sqrt(real_Gx**2 + real_Gy**2)

    # similarity
    rec_S_V = (2*real_GM*rec_GM+160)/(real_GM**2+rec_GM**2+160)
    rec_S_M = (2*rec_J_YMN[:,:,1]*real_I_YMN[:,:,1]+130)/(rec_J_YMN[:,:,1]**2+real_I_YMN[:,:,1]**2+130)
    rec_S_N = (2*rec_J_YMN[:,:,2]*real_I_YMN[:,:,2]+130)/(rec_J_YMN[:,:,2]**2+real_I_YMN[:,:,2]**2+130)
    rec_S_R = (rec_S_M*rec_S_N).reshape([recH,recW])

    refine_S_V = (2*real_GM*refine_GM+160)/(real_GM**2+refine_GM**2+160)
    refine_S_M = (2*refine_J_YMN[:,:,1]*real_I_YMN[:,:,1]+130)/(refine_J_YMN[:,:,1]**2+real_I_YMN[:,:,1]**2+130)
    refine_S_N = (2*refine_J_YMN[:,:,2]*real_I_YMN[:,:,2]+130)/(refine_J_YMN[:,:,2]**2+real_I_YMN[:,:,2]**2+130)
    refine_S_R = (refine_S_M*refine_S_N).reshape([recH,recW])


    rec_S = rec_S_R*np.power(rec_S_V, 0.4)
    refine_S = refine_S_R*np.power(refine_S_V, 0.4)


    fuseWeight = np.exp(rec_S)/(np.exp(rec_S)+np.exp(refine_S))
    fuseWeightMap = fuseWeight.reshape([recH,recW,1]).repeat(3,axis=2)

    fuse_J = rec_J*fuseWeightMap + refine_J*(1-fuseWeightMap)
    return fuse_J


def BCCR_dehaze(x):
    BCCR_img_dehazer = image_dehazer()
    images = x.cpu().permute(0, 2, 3, 1).float().numpy()
    images = (images * 255).round().astype("uint8")

    J_BCCR, T_BCCR = BCCR_img_dehazer.remove_haze(images[0])
    J_BCCR_tensor = (J_BCCR / 255.0)
    J_BCCR_tensor = torch.from_numpy(J_BCCR_tensor.transpose(2, 0, 1)).unsqueeze(0)
    T_BCCR_tensor = torch.from_numpy(T_BCCR).unsqueeze(0).unsqueeze(0)
    return J_BCCR, T_BCCR, J_BCCR_tensor, T_BCCR_tensor

class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y

class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y


class MFM(nn.Module):
    def __init__(self, dim, height=2, reduction=8):
        super(MFM, self).__init__()

        self.height = height
        d = max(int(dim/reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(d, dim*height, 1, bias=False)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_feats):
        B, C, H, W = in_feats[0].shape

        in_feats = torch.cat(in_feats, dim=1)
        in_feats = in_feats.view(B, self.height, C, H, W)

        feats_sum = torch.sum(in_feats, dim=1)
        attn = self.mlp(self.avg_pool(feats_sum))
        attn = self.softmax(attn.view(B, self.height, C, 1, 1))

        out = torch.sum(in_feats*attn, dim=1)
        return out



class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y


class guidance_BCCR_DCP_image(nn.Module):
    def __init__(self, input_channel, hidden_channel, output_channel, ):
        super(guidance_BCCR_DCP_image, self).__init__()
        # self.mlp = nn.Sequential(
        #     nn.Conv2d(hidden_channel, hidden_channel, 1, bias=False),
        #     nn.ReLU(),
        #     nn.Conv2d(hidden_channel, input_channel, 1, bias=False),
        # )
        self.input_channel = input_channel
        self.hidden_channel = hidden_channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.point_wise_conv = nn.Conv2d(hidden_channel, hidden_channel, 1)
        self.input_p_conv = nn.Conv2d(input_channel, hidden_channel, 1)
        self.relu = nn.ReLU()

        self.g_conv1 = nn.Sequential(
            nn.Conv2d(input_channel, hidden_channel, kernel_size=1),
            nn.Conv2d(hidden_channel, hidden_channel, kernel_size=3, padding=1, groups=hidden_channel, padding_mode='reflect'),
            nn.ReLU(inplace=True),
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(hidden_channel, output_channel, kernel_size=1),
            # nn.Conv2d(output_channel, output_channel, kernel_size=3, padding=1, groups=output_channel, padding_mode='reflect'),
            # nn.Sigmoid(),
        )



    def forward(self, dcp_image, bccr_image, real_haze):
        dcp_image_ = dcp_image
        bccr_image_ = bccr_image
        real_haze_ = real_haze
        real_haze = self.g_conv1(real_haze)
        dcp_image = self.relu(self.input_p_conv(dcp_image))
        bccr_image = self.relu(self.input_p_conv(bccr_image))

        real_haze_dcp = self.out_conv(real_haze + dcp_image)
        real_haze_bccr = self.out_conv(real_haze + bccr_image)

        dcp_output = dcp_image_ * real_haze_dcp
        bccr_output = bccr_image_ * real_haze_bccr

        return dcp_output, bccr_output




class fuse_dcp_bccr(nn.Module):
    def __init__(self, input_channel, hidden_channel):
        super(fuse_dcp_bccr, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(input_channel * 2, hidden_channel, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden_channel, input_channel * 2, 1, bias=False),
        )
        self.softmax = nn.Softmax(dim=1)
        self.gate_conv = nn.Conv2d(input_channel * 4, 4, 1)



    def forward(self, dcp, bccr):
        input = torch.cat((dcp, bccr), dim=1)
        input_GAP = self.avg_pool(input)
        input_GAP = self.mlp(input_GAP)
        soft_input_GAP = self.softmax(input_GAP)
        middle_output = input * soft_input_GAP + input
        output = self.gate_conv(torch.cat((input, middle_output), dim=1))
        final_output = output[:,0, :, :] * dcp + output[:,1, :, :] * bccr + middle_output[:, :3, :, :] * output[:,2, :, :] + middle_output[:, -3:, :, :] * output[:, 3, :, :]
        return final_output

class fuse_block(nn.Module):
    def __init__(self, input_channel=3, hidden_channel=6, output_channel=3):
        super(fuse_block, self).__init__()
        self.guide_block = guidance_BCCR_DCP_image(input_channel, hidden_channel, output_channel)
        self.fuse_block = fuse_dcp_bccr(input_channel, hidden_channel)

    def forward(self, dcp, bccr, real_haze):
        dcp_output, bccr_output = self.guide_block(dcp, bccr, real_haze)

        final_output = self.fuse_block(dcp_output, bccr_output)

        return final_output

def haze_density_estimate(img):
    guided_filter = GuidedFilter(r=15, eps=1e-3)

    gray_img_tensor = 0.2989 * img[:, 0, :, :] + 0.5870 * img[:, 1, :, :] + 0.1140 * img[:, 2, :, :]

    gray_img = gray_img_tensor.cpu().numpy().squeeze(0)

    sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3).astype(np.float32)
    sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3).astype(np.float32)
    # 计算最终的Sobel梯度
    sobel_gradient = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    sobel_gradient = torch.from_numpy(sobel_gradient).unsqueeze(0).unsqueeze(0)

    saturation, value = rgb_to_hsv(img)
    saturation_normalize = (saturation - saturation.min()) / (saturation.max() - saturation.min())
    saturation_normalize = get_near_mean(saturation_normalize, kernel_size=15)
    value_normalize = (value - value.min()) / (value.max() - value.min())
    value_normalize = get_near_mean(value_normalize, kernel_size=15)
    sobel_gradient_normalize = (sobel_gradient - sobel_gradient.min()) / (sobel_gradient.max() - sobel_gradient.min())
    sobel_gradient_normalize = get_near_mean(sobel_gradient_normalize, kernel_size=15)
    theat_1 = torch.tensor(0.959710)
    theat_2 = torch.tensor(-0.780245)
    diff_l_s = 0.121779 + theat_1 * value_normalize + theat_2 * saturation_normalize - 0.2 * sobel_gradient_normalize
    diff_l_s = torch.clamp(diff_l_s, 0.05, 0.95)
    density = guided_filter(gray_img_tensor, diff_l_s)
    # mean = density.mean()

    return density

def get_near_mean(img, kernel_size):
    import torch.nn.functional as F
    B, C, H, W = img.shape
    stride = 1
    padding = kernel_size // 2
    dilation = 1
    output_tensor = F.unfold(img, kernel_size, dilation, padding, stride)
    output_tensor = output_tensor.mean(dim=1)

    # 将输出tensor调整为原始图像的形状
    output_tensor = output_tensor.view(B, C, H, W)

    return output_tensor


def rgb_to_hsv(img):
    eps = 1e-8
    saturation = (img.max(1)[0] - img.min(1)[0]) / (img.max(1)[0] + eps)
    saturation[img.max(1)[0] == 0] = 0
    value = img.max(1)[0]
    saturation = saturation.unsqueeze(1)
    value = value.unsqueeze(1)
    return saturation, value

def haze_control(img, haze_density):
    haze_density = F.interpolate(haze_density, size=img.shape[-2:], mode='bilinear')
    B, C, H, W = haze_density.shape
    haze_density_ = haze_density
    img_guided = torch.mul(img, haze_density)
    img_guided = img_guided + img
    return img
class down_haze_density_block(nn.Module):
    def __init__(self, input_channel=3, hidden_channel=128, output_channel=[]):
        super(down_haze_density_block, self).__init__()
        self.conv_in = nn.Conv2d(input_channel, hidden_channel, 1)
        self.conv_in_pos= nn.Conv2d(hidden_channel, hidden_channel, 1)
        self.conv_0 = nn.Conv2d(hidden_channel, output_channel[0], 3, 2, 1)
        self.relu = nn.ReLU()
        self.conv_1 = nn.Conv2d(output_channel[0], output_channel[1], 3, 2, 1)
        self.conv_2 = nn.Conv2d(output_channel[1], output_channel[2], 3, 2, 1)
        self.pos_0 = nn.Conv2d(output_channel[0], output_channel[0], 1)
        self.pos_1 = nn.Conv2d(output_channel[1], output_channel[1], 1)
        self.pos_2 = nn.Conv2d(output_channel[2], output_channel[2], 1)
    def forward(self, haze_density):
        haze_density_ = self.conv_in_pos(self.relu(self.conv_in(haze_density)))
        haze_density_1 = self.pos_0(self.relu(self.conv_0(haze_density_)))
        haze_density_2 = self.pos_1(self.relu(self.conv_1(haze_density_1)))
        haze_density_3 = self.pos_2(self.relu(self.conv_2(haze_density_2)))

        return [haze_density, haze_density_1, haze_density_2, haze_density_3]


# class haze_density_control(nn.Module):
#     def __init__(self, input_channel, output_channel):
#         super(haze_density_control, self).__init__()
#         self.point_conv = nn.Conv2d(input_channel, output_channel, 1)
#         self.point_conv_in = nn.Conv2d(input_channel, output_channel, 1)
#         self.softmax = nn.Softmax(dim=1)
#         self.ca = CALayer(input_channel)
#         self.mlp = nn.Sequential(
#             nn.Conv2d(input_channel , output_channel, 1, bias=False),
#             nn.ReLU(),
#             nn.Conv2d(output_channel, input_channel , 1, bias=False),
#         )
#
#     def forward(self, img, haze_density):
#
#         haze_density = F.interpolate(haze_density, size=img.shape[-2:], mode='bilinear')
#         B, C, H, W = haze_density.shape
#         haze_density_ = haze_density
#         img_ = img
#         img = self.point_conv_in(img)
#         img = self.ca(img)
#         # haze_density = self.point_conv(haze_density)
#         haze_density = haze_density.permute(0, 2, 3, 1).reshape(-1, H*W*C)
#         haze_density = self.softmax(haze_density) + 1e-7
#         haze_density = haze_density.reshape(B, H, W, C).permute(0, 3, 1, 2)
#         img_guided = torch.mul(img, haze_density)
#         img_guided = img_guided + img_
#
#         return img_guided
def haze_softmax(img, haze_density):
    softmax = nn.Softmax(dim=1)
    haze_density_ = haze_density
    B, C, H, W = img.shape
    img_zero = torch.zeros_like(img)


    # haze_density = nn.AdaptiveAvgPool2d(8)(haze_density)
    # B, C, H, W = haze_density.shape
    # haze_density = haze_density.permute(0, 2, 3, 1).reshape(-1, H * W * C)
    # haze_density = softmax(haze_density) + 1e-7
    # haze_density = haze_density.reshape(B, H, W, C).permute(0, 3, 1, 2)
    # haze_density = haze_density.repeat_interleave(64, dim=2)
    # haze_density = haze_density.repeat_interleave(64, dim=3)
    img_guided = torch.mul(img, haze_density)
    img_guided = torch.clamp(img_guided, 0, 1)
    return img_guided






if __name__ == '__main__':
    import glob
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    img_dir = '/data/lanyunwei/my_experiment_results/O-HAZE/train_A'
    # img_pth = '/data/lanyunwei/my_experiment_results/O-HAZE/train_A/16_outdoor_GT.jpg'
    img_pth = '/data/lanyunwei/my_experiment_results/O-HAZE/train_B/16_outdoor_hazy.jpg'
    img_name = img_pth.split('/')[-1]
    input_transfer = Compose([
        ToTensor(),
        # torchvision.transforms.Resize(512),
    ])
    output_dir = '/output/'
    os.makedirs(output_dir, exist_ok=True)
    save_dir = os.path.join(output_dir, f'GT_{img_name}')
    img = Image.open(img_pth).convert('RGB')
    img = input_transfer(img)
    img = img.unsqueeze(0).to(device)
    haze_density = haze_density_estimate(img)
    # output = haze_softmax(img, haze_density)
    # down_block = down_haze_density_block(output_channel=[56, 128, 46])
    # output = down_block(output)
    # print(output)
    output_mul = torch.clamp(torch.mul(img, haze_density), 0, 1)
    # out = ToPILImage()(output[0])
    # out.save('/output/out.jpg')
    # list = []
    # image_extensions = ['*.jpg', '*.jpeg', '*.png']
    # for extension in image_extensions:
    #     # 使用glob模块获取所有匹配的图片文件
    #     list += glob.glob(os.path.join(img_dir, extension))
    # # img_name = img_pth.split('/')[-1]
    # # output_dir = '/output/'
    # # save_dir = os.path.join(output_dir, img_name)
    # # img = Image.open(img_pth).convert('RGB')
    # input_transfer = Compose([
    #     ToTensor(),
    #     torchvision.transforms.Resize(512),
    # ])
    output_transfer = Compose([
        ToPILImage()
    ])
    # for img_pth in list:
    #     img_name = img_pth.split('/')[-1]
    #     output_dir = '/output/density/'
    #     os.makedirs(output_dir, exist_ok=True)
    #     save_dir = os.path.join(output_dir, f'GT_{img_name}')
    #     img = Image.open(img_pth).convert('RGB')
    #     img = input_transfer(img)
    #     img = img.unsqueeze(0).to(device)
    #     out = haze_density_estimate(img)
    #     out = ToPILImage()(out[0])
    #     out.save(save_dir)



    # img = torch.flip(img, [1])
    # dcp_out = haze_density_estimate(img)
    # out = haze_density_estimate(img, dcp_out)





    # dcp_out = ToPILImage()(t[0])
    # dcp_out.save('/output/dcp.jpg')
    # print(out)

    # a = torch.randn(256, 256, 3)
    # b = torch.randn(256, 256, 3)
    # c = torch.randn(256, 256, 3)
    # d = fuse_images(a, b, c)

    import matplotlib.colors as mcolors
    # BCCR_dehaze(img)
    # fuse = guidance_BCCR_DCP_image(3, 6, 3)
    # a = torch.randn(1, 3, 256, 256)
    # b = torch.randn(1, 3, 256, 256)
    # c = torch.randn(1, 3, 256, 256)
    # d = fuse(a, b, c)
    # print(img.shape)
    # J_DCP, T_DCP, A = dark_channel_generate(img)
    # J_BCCR, T_BCCR, J_BCCR_tensor, T_BCCR_tensor = BCCR_dehaze(img)
    # J_BCCR_tensor = torch.clamp(J_BCCR_tensor, 0, 1)
    # J_BCCR_tensor = output_transfer(J_BCCR_tensor[0])
    # J_BCCR_tensor.save('/output/bccr_j.jpg')
    # T_BCCR_tensor = torch.clamp(T_BCCR_tensor, 0, 1)
    T_BCCR_tensor = output_transfer(haze_density[0])
    T_BCCR_tensor.save('/output/bccr_t.jpg')
    # J_BCCR = Image.fromarray(J_BCCR)
    # J_BCCR.save('/output/b_J.jpg')
    # T_BCCR = Image.fromarray(T_BCCR, mode='L')
    # T_BCCR.save('/output/b_T.jpg')
    # t_slide = output_transfer(T_DCP[0])
    # t_slide.save(save_dir)




    # dark_channel = get_bright_channel(img)
    # bright_channel = get_bright_channel(img)
    # A_dark = get_atmosphere_dark(img, dark_channel)
    # A_bright = get_atmosphere_bright(img, bright_channel)
    # J_DCP, T_DCP = dark_channel_generate(img)
    # J_DCP = torch.clamp(J_DCP, 0, 1)
    # J_DCP = output_transfer(J_DCP[0])
    # J_DCP.save(save_dir)
    # t_slide = bright_channel_generate(img)
    # t_slide = output_transfer(t_slide[0])
    # t_slide.save(save_dir)
    # image_tensor = torch.randn(1, 1, 4, 4)
    #
    # # 使用unfold函数选择大小为5的邻域
    # kernel_size = 3
    # stride = 1
    # padding = 1
    # dilation = 1
    # output_tensor = F.unfold(image_tensor, kernel_size, dilation, padding, stride)
    #
    # # 对邻域内的像素求均值
    # output_tensor = output_tensor.mean(dim=1)
    #
    # # 将输出tensor调整为原始图像的形状
    # output_tensor = output_tensor.view(1, 1, 4, 4)
    # print(output_tensor)







