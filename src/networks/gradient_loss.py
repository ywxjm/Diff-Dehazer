import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientLoss(nn.Module):
    def __init__(self, operator="Sobel", channel_mean=True):
        r"""
       :param operator: in ['Sobel', 'Prewitt','Roberts','Scharr']
       :param channel_mean: 是否在通道维度上计算均值
       """
        super(GradientLoss, self).__init__()
        assert operator in ['Sobel', 'Prewitt', 'Roberts', 'Scharr'], "Unsupported operator"
        self.channel_mean = channel_mean

        self.operators = {
            "Sobel": {
                'x': torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float),
                'y': torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=torch.float)
            },
            "Prewitt": {
                'x': torch.tensor([[[[1, 0, -1], [1, 0, -1], [1, 0, -1]]]], dtype=torch.float),
                'y': torch.tensor([[[[-1, -1, -1], [0, 0, 0], [1, 1, 1]]]], dtype=torch.float)
            },
            "Roberts": {
                'x': torch.tensor([[[[1, 0], [0, -1]]]], dtype=torch.float),
                'y': torch.tensor([[[[0, -1], [1, 0]]]], dtype=torch.float)
            },
            "Scharr": {
                'x': torch.tensor([[[[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]]]], dtype=torch.float),
                'y': torch.tensor([[[[-3, 10, -3], [0, 0, 0], [3, 10, 3]]]], dtype=torch.float)
            },
        }
        self.op_x = self.operators[operator]['x']
        self.op_y = self.operators[operator]['y']

    def gradients(self, img_tensor):
        op_x, op_y = self.op_x.to('cuda'), self.op_y.to('cuda')
        if self.channel_mean:
            img_tensor = img_tensor.mean(dim=1, keepdim=True)
            groups = 1
        else:
            groups = img_tensor.shape[1]
            op_x = op_x.repeat(groups, 1, 1, 1)
            op_y = op_y.repeat(groups, 1, 1, 1)
        grad_x = F.conv2d(img_tensor, op_x, groups=groups)
        grad_y = F.conv2d(img_tensor, op_y, groups=groups)
        return grad_x, grad_y

    def forward(self, img1, img2):
        grad_x1, grad_y1 = self.gradients(img1)
        grad_x2, grad_y2 = self.gradients(img2)
        diff_x = torch.abs(grad_x1 - grad_x2)
        diff_y = torch.abs(grad_y1 - grad_y2)
        total_loss = torch.mean(diff_x + diff_y)
        return total_loss



class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(3, 1, 1, 1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = nn.L1Loss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)
        down        = filtered[:,:,::2,::2]
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4
        filtered    = self.conv_gauss(new_filter)
        diff = current - filtered
        return diff

    def forward(self, input, target):
        loss = self.loss(self.laplacian_kernel(input), self.laplacian_kernel(target))
        return loss


# 定义一个多尺度损失函数
class MultiScaleLoss(nn.Module):
    def __init__(self, scales):
        super(MultiScaleLoss, self).__init__()
        self.scales = scales
        self.criterion = nn.L1Loss()

    def forward(self, output, target):
        loss = 0
        for scale in self.scales:
            output_scaled = nn.functional.interpolate(output, scale_factor=scale, mode='bilinear', align_corners=False)
            target_scaled = nn.functional.interpolate(target, scale_factor=scale, mode='bilinear', align_corners=False)
            loss += self.criterion(output_scaled, target_scaled)
        return loss / len(self.scales)

# 定义输入和目标图像


# 定义多尺度损失函数


if __name__ == '__main__':
    input_image = torch.randn(1, 3, 256, 256)
    target_image = torch.randn(1, 3, 256, 256)

    # 定义模型输出
    output_image = torch.randn(1, 3, 256, 256)
    multi_scale_loss = MultiScaleLoss([0.5, 1.0, 2.0])
#     x = torch.randn(5, 3, 32, 64).to('cuda')
#     x = x.to(torch.float32)
#
#     # print(EdgeLoss()(x, x ** 2))