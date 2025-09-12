import torch, os
from .blocks import FeatureFusionBlock_custom, _make_scratch, _make_pretrained_efficientnet_lite3
from .blocks import Interpolate
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            # print(m.name)
            # if classname.find('pretrained') != -1:
            #     print(classname)
            #     return
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)


class HazeRemovalNet(BaseNetwork):
    def __init__(self, base_channel_nums, init_weights=True, path=None, min_beta=0.04, max_beta=0.2, min_d=0.3, max_d=5, use_dc_A=False):
        super(HazeRemovalNet, self).__init__()
        self.transmission_estimator = TransmissionEstimator()

        # norm_type = 'batch'
        # act_type = 'leakyrelu'
        # mode = 'CNA'
        use_spectral_norm = False

        self.MIN_BETA=min_beta
        self.MAX_BETA=max_beta
        self.MIN_D = min_d
        self.MAX_D = max_d
        self.use_dc_A = True if use_dc_A == 1 else False

        backbone = "efficientnet_lite3"
        non_negative = True
        exportable = True
        align_corners = True
        blocks = {'expand': True}

        features = base_channel_nums

        use_pretrained = False if os.path.exists(path) else True

        # self.channels_last = channels_last
        self.blocks = blocks
        self.backbone = backbone

        self.groups = 1

        features1 = features
        features2 = features
        features3 = features
        features4 = features
        self.expand = False
        if "expand" in self.blocks and self.blocks['expand'] == True:
            self.expand = True
            features1 = features
            features2 = features * 2
            features3 = features * 4
            features4 = features * 8


        self.scratch = _make_scratch([32, 48, 136, 384], features, groups=self.groups,expand=self.expand)


        self.scratch.activation = nn.ReLU(False)

        self.scratch.refinenet4 = FeatureFusionBlock_custom(features4, self.scratch.activation, deconv=False, bn=False,
                                                            expand=self.expand, align_corners=align_corners)
        self.scratch.refinenet3 = FeatureFusionBlock_custom(features3, self.scratch.activation, deconv=False, bn=False,
                                                            expand=self.expand, align_corners=align_corners)
        self.scratch.refinenet2 = FeatureFusionBlock_custom(features2, self.scratch.activation, deconv=False, bn=False,
                                                            expand=self.expand, align_corners=align_corners)
        self.scratch.refinenet1 = FeatureFusionBlock_custom(features1, self.scratch.activation, deconv=False, bn=False,
                                                            align_corners=align_corners)

        self.scratch.output_conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=0, groups=self.groups),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.ReflectionPad2d(1),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=0),
            self.scratch.activation,
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),

        )


        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=[1,1])
        self.final_conv_beta_1 = conv_block(in_nc=32 +48 +136 +384, out_nc=2*base_channel_nums, kernel_size=1, act_type=None, norm_type=None, use_spectral_norm=use_spectral_norm)
        self.final_conv_beta_2 = conv_block(in_nc=2*base_channel_nums, out_nc=1, kernel_size=1, act_type=None, norm_type=None, use_spectral_norm=use_spectral_norm)


        if init_weights:
            self.init_weights('xaiver')

        self.pretrained =_make_pretrained_efficientnet_lite3(use_pretrained, exportable=exportable)

    def forward_get_A(self, x): # output A: N,3,1,1
        if self.use_dc_A:
            A = self.transmission_estimator.get_atmosphere_light_new(x)
        else:
            A = x.max(dim=3)[0].max(dim=2,keepdim=True)[0].unsqueeze(3)

        return A

    def forward(self, x_0, require_paras=False, use_guided_filter=False):

        layer_1 = self.pretrained.layer1(x_0)
        layer_2 = self.pretrained.layer2(layer_1)
        layer_3 = self.pretrained.layer3(layer_2)
        layer_4 = self.pretrained.layer4(layer_3)

        layer_1_beta = F.adaptive_avg_pool2d(layer_1, [1,1]).detach()
        layer_2_beta = F.adaptive_avg_pool2d(layer_2, [1, 1]).detach()
        layer_3_beta = F.adaptive_avg_pool2d(layer_3, [1, 1]).detach()
        layer_4_beta = F.adaptive_avg_pool2d(layer_4, [1, 1]).detach()

        beta = self.final_conv_beta_1(torch.cat([layer_1_beta, layer_2_beta, layer_3_beta, layer_4_beta], dim=1))
        beta = self.final_conv_beta_2(beta)


        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)

        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)



        t = self.scratch.output_conv(path_1)

        beta = self.MIN_BETA + (self.MAX_BETA-self.MIN_BETA)*(torch.tanh(beta) + 1) / 2


        t = ((torch.tanh(t) + 1) / 2)
        t = t.clamp(0.05,0.95)

        if use_guided_filter:
            t = self.transmission_estimator.get_refined_transmission(x_0,t)

        A = self.forward_get_A(x_0)
        d = torch.log(t)/(-beta)

        if require_paras:
            return ((x_0-A)/t + A).clamp(0,1), d, beta
        else:
            return ((x_0-A)/t + A).clamp(0,1), t


class DepthEstimationNet(BaseNetwork):
    def __init__(self, base_channel_nums=48, min_d=0.3, max_d=10,
                 path="/data/code/", init_weights=True):
        super(DepthEstimationNet, self).__init__()

        self.MIN_D = min_d
        self.MAX_D = max_d

        backbone = "efficientnet_lite3"
        exportable = True
        align_corners = True
        blocks = {'expand': True}

        features = base_channel_nums

        use_pretrained = False if os.path.exists(path) else True
        # use_pretrained = True
        self.blocks = blocks
        self.backbone = backbone

        self.groups = 1

        features1 = features
        features2 = features
        features3 = features
        features4 = features
        self.expand = False
        if "expand" in self.blocks and self.blocks['expand'] == True:
            self.expand = True
            features1 = features
            features2 = features * 2
            features3 = features * 4
            features4 = features * 8

        self.scratch = _make_scratch([32, 48, 136, 384], features, groups=self.groups, expand=self.expand)

        self.scratch.activation = nn.ReLU(False)

        self.scratch.refinenet4 = FeatureFusionBlock_custom(features4, self.scratch.activation, deconv=False,
                                                            bn=False,
                                                            expand=self.expand, align_corners=align_corners)
        self.scratch.refinenet3 = FeatureFusionBlock_custom(features3, self.scratch.activation, deconv=False,
                                                            bn=False,
                                                            expand=self.expand, align_corners=align_corners)
        self.scratch.refinenet2 = FeatureFusionBlock_custom(features2, self.scratch.activation, deconv=False,
                                                            bn=False,
                                                            expand=self.expand, align_corners=align_corners)
        self.scratch.refinenet1 = FeatureFusionBlock_custom(features1, self.scratch.activation, deconv=False,
                                                            bn=False,
                                                            align_corners=align_corners)

        self.scratch.output_conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=0, groups=self.groups),
            Interpolate(scale_factor=2, mode="bilinear"),
            nn.ReflectionPad2d(1),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=0),
            self.scratch.activation,
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
        )

        if init_weights:
            self.init_weights('xaiver')

        self.pretrained = _make_pretrained_efficientnet_lite3(use_pretrained, exportable=exportable)

    def forward(self, x):
        layer_1 = self.pretrained.layer1(x)
        layer_2 = self.pretrained.layer2(layer_1)
        layer_3 = self.pretrained.layer3(layer_2)
        layer_4 = self.pretrained.layer4(layer_3)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv(path_1)
        out = ((torch.tanh(out) + 1) / 2)
        # out = self.MIN_D + out * (self.MAX_D - self.MIN_D)

        return out





class TransmissionEstimator(nn.Module):
    def __init__(self, width=15,):
        super(TransmissionEstimator, self).__init__()
        self.width = width
        self.t_min = 0.2
        self.alpha = 2.5
        self.A_max = 220.0/255
        self.omega=0.95
        self.p = 0.001
        self.max_pool = nn.MaxPool2d(kernel_size=width,stride=1)
        self.max_pool_with_index = nn.MaxPool2d(kernel_size=width, return_indices=True)
        self.guided_filter = GuidedFilter(r=40,eps=1e-3)


    def get_dark_channel(self, x):
        x = torch.min(x, dim=1, keepdim=True)[0]
        x = F.pad(x, (self.width//2, self.width//2,self.width//2, self.width//2), mode='constant', value=1)
        x = -(self.max_pool(-x))
        return x

    def get_atmosphere_light(self,I,dc):
        n,c,h,w = I.shape
        flat_I = I.view(n,c,-1)
        flat_dc = dc.view(n,1,-1)
        searchidx = torch.argsort(flat_dc, dim=2, descending=True)[:,:,:int(h*w*self.p)]
        searchidx = searchidx.expand(-1,3,-1)
        searched = torch.gather(flat_I,dim=2, index=searchidx)
        return torch.max(searched, dim=2 ,keepdim=True)[0].unsqueeze(3)

    def get_transmission(self, I, A):
        return 1-self.omega* self.get_dark_channel(I/A)

    def get_refined_transmission(self, I, rawt):
        I_max = torch.max(I.contiguous().view(I.shape[0],-1), dim=1, keepdim=True)[0].unsqueeze(2).unsqueeze(3)
        I_min = torch.min(I.contiguous().view(I.shape[0],-1), dim=1, keepdim=True)[0].unsqueeze(2).unsqueeze(3)
        normI = (I - I_min)/(I_max-I_min)
        refinedT = self.guided_filter(normI, rawt)


        return refinedT

    def get_radiance(self,I, A, t):
        return (I-A)/t + A



    def get_depth(self, I):
        I_dark = self.get_dark_channel(I)

        A = self.get_atmosphere_light(I, I_dark)
        A[A>self.A_max] = self.A_max
        rawT = self.get_transmission(I, A)

        # print(I)

        refinedT = self.get_refined_transmission(I, rawT)
        return refinedT

    def get_atmosphere_light_new(self, I):
        I_dark = self.get_dark_channel(I)
        A = self.get_atmosphere_light(I, I_dark)
        A[A > self.A_max] = self.A_max
        return A


class GuidedFilter(nn.Module):
    def __init__(self, r, eps=1e-8):
        super(GuidedFilter, self).__init__()

        self.r = r
        self.eps = eps
        self.boxfilter = BoxFilter(r)


    def forward(self, x, y):
        n_x, c_x, h_x, w_x = x.size()
        n_y, c_y, h_y, w_y = y.size()

        assert n_x == n_y
        #assert c_x == 1 or c_x == c_y
        assert h_x == h_y and w_x == w_y
        assert h_x > 2 * self.r + 1 and w_x > 2 * self.r + 1

        # N
        N = self.boxfilter(Variable(x.data.new().resize_((1, 1, h_x, w_x)).fill_(1.0)))

        # mean_x
        mean_x = self.boxfilter(x) / N
        # mean_y
        mean_y = self.boxfilter(y) / N
        # cov_xy
        cov_xy = self.boxfilter(x * y) / N - mean_x * mean_y
        # var_x
        var_x = self.boxfilter(x * x) / N - mean_x * mean_x

        # A
        A = cov_xy / (var_x + self.eps)
        # b
        b = mean_y - A * mean_x

        # mean_A; mean_b
        mean_A = self.boxfilter(A) / N
        mean_b = self.boxfilter(b) / N

        return torch.mean(mean_A * x + mean_b, dim=1 ,keepdim=True)


class BoxFilter(nn.Module):
    def __init__(self, r):
        super(BoxFilter, self).__init__()

        self.r = r

    def forward(self, x):
        assert x.dim() == 4

        return diff_y(diff_x(x.cumsum(dim=2), self.r).cumsum(dim=3), self.r)


def diff_x(input, r):
    assert input.dim() == 4

    left   = input[:, :,         r:2 * r + 1]
    middle = input[:, :, 2 * r + 1:         ] - input[:, :,           :-2 * r - 1]
    right  = input[:, :,        -1:         ] - input[:, :, -2 * r - 1:    -r - 1]

    output = torch.cat([left, middle, right], dim=2)

    return output

def diff_y(input, r):
    assert input.dim() == 4

    left   = input[:, :, :,         r:2 * r + 1]
    middle = input[:, :, :, 2 * r + 1:         ] - input[:, :, :,           :-2 * r - 1]
    right  = input[:, :, :,        -1:         ] - input[:, :, :, -2 * r - 1:    -r - 1]

    output = torch.cat([left, middle, right], dim=3)

    return output