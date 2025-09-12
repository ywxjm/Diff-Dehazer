import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.image_processor import VaeImageProcessor
from PIL import Image
import torchvision.transforms.functional as FF

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


class DCP_Generator(nn.Module):
    def __init__(self, neighborhood_size=5, omega=0.8):
        super(DCP_Generator, self).__init__()

        self.guid_filter = GuidedFilter(r=15, eps=1e-3)
        self.neighborhood_size = neighborhood_size
        self.omega = omega

    def get_dark_channel(self, img, neighborhood_size):
        shape = img.shape
        img_min, _ = torch.min(img, dim=1)
        if len(shape) == 4:
            dark_channel = -F.max_pool2d(-img_min, kernel_size=neighborhood_size, stride=1,
                                         padding=neighborhood_size // 2)

            # 归一化处理
            # dark_image = (dark_channel - dark_channel.min()) / (dark_channel.max() - dark_channel.min())

        else:
            raise NotImplementedError('get_tensor_dark_channel is only for 4-d tensor [N*C*H*W]')

        dark_img = torch.unsqueeze(dark_channel, dim=1)
        return dark_img

    def forward(self, img):
        h, w = img.shape[2:]
        imsz = h * w
        # 要查找的是暗通道中前0.1%的值
        numpx = max(imsz // 1000, 1)
        # 找到暗通道的索引，弄成[batch, 3, numpx]，因为要匹配三个通道，所以需要expand
        # dark = torch.min(img, dim=1, keepdim=True)[0]
        dark = self.get_dark_channel(img, neighborhood_size=5)
        indices = torch.topk(dark.view(-1, imsz), k=numpx, dim=1)[1].view(-1, 1, numpx).expand(-1, 3, -1)
        # 用上述索引匹配原图中的3个通道，并求其平均值
        a = (torch.gather(img.view(-1, 3, imsz), 2, indices).sum(2) / numpx).view(-1, 3, 1, 1)

        # 代公式算tx
        tx = 1 - self.omega * self.get_dark_channel(img / a, neighborhood_size=5)

        guidance = 0.2989 * img[:, 0, :, :] + 0.5870 * img[:, 1, :, :] + 0.1140 * img[:, 2, :, :]
        guidance = torch.unsqueeze(guidance, dim=1)
        t_guid = self.guid_filter(guidance, tx)
        # 代公式算jx
        return (img - a) / torch.clamp_min(t_guid, 0.1) + a, t_guid, dark, a

if __name__ == '__main__':
    vae_img_processor = VaeImageProcessor(vae_scale_factor=8, do_convert_rgb=True,
                                          do_normalize=False)
    img_pth = '/data/lanyunwei/my_experiment_results/unpaired_dehazing_dataset_simple/train_B/YT_Google_624.jpeg'
    img = Image.open(img_pth).convert('RGB')
    img = vae_img_processor.pil_to_numpy(img)
    img = vae_img_processor.numpy_to_pt(img).to('cuda')
    img = vae_img_processor.resize(img, 128, 128)
    img = FF.normalize(img, mean=[0.5], std=[0.5])
    dcp = DCP_Generator(5, 0.75).to('cuda')
    j, t_guid, dark, depth = dcp(img)
    J_DCP = vae_img_processor.pt_to_numpy((j + 1) / 2.)
    J_DCP = vae_img_processor.numpy_to_pil(J_DCP)
    J_DCP[0].save('/output/J_DCP.png')
    T_DCP = vae_img_processor.pt_to_numpy((t_guid + 1) / 2.)
    T_DCP = vae_img_processor.numpy_to_pil(T_DCP)
    T_DCP[0].save('/output/T_DCP.png')
    dark_img = vae_img_processor.pt_to_numpy(dark)
    dark_img = vae_img_processor.numpy_to_pil(dark_img)
    dark_img[0].save('/output/dark.png')
    depth = vae_img_processor.pt_to_numpy(depth)
    depth = vae_img_processor.numpy_to_pil(depth)
    depth[0].save('/output/depth.png')