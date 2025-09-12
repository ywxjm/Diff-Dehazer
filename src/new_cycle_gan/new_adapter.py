import torch.nn as nn
import torch.nn.functional as F
import torch
class Mlp(nn.Module):
    def __init__(self, network_depth, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.network_depth = network_depth
        self.mlp = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1),
            nn.ReLU(True),
            nn.Conv2d(hidden_features, out_features, 1)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            gain = (8 * self.network_depth) ** (-1/4)
            fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            trunc_normal_(m.weight, std=std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        return self.mlp(x)


class Enhanced_block(nn.Module):
    def __init__(self):
        super(Enhanced_block, self).__init__()

        self.relu=nn.LeakyReLU(0.2, inplace=True)

        self.tanh=nn.Tanh()

        self.conv1010 = nn.Conv2d(128, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1020 = nn.Conv2d(128, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1030 = nn.Conv2d(256, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1040 = nn.Conv2d(512, 1, kernel_size=1,stride=1,padding=0)  # 1mm

        self.refine3= nn.Conv2d(3+4, 3, kernel_size=3,stride=1,padding=1)

        self.upsample = F.upsample_nearest

        self.batch1 = nn.InstanceNorm2d(100, affine=True)

    def forward(self, x):
        x_0, x_1, x_2, x_3 = x[0], x[1], x[2], x[3]
        shape_out = x_0.data.size()
        # print(shape_out)
        shape_out = shape_out[2:4]


        x1010 = self.upsample(self.relu(self.conv1010(x_0)),size=shape_out)
        x1020 = self.upsample(self.relu(self.conv1020(x_1)),size=shape_out)
        x1030 = self.upsample(self.relu(self.conv1030(x_2)),size=shape_out)
        x1040 = self.upsample(self.relu(self.conv1040(x_3)),size=shape_out)

        dehaze = torch.cat((x1010, x1020, x1030, x1040, dehaze), 1)
        dehaze= self.tanh(self.refine3(dehaze))

        return dehaze



class CrossAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(CrossAttention, self).__init__()
        self.linear_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.linear_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.linear_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.linear_out = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x1, x2):
        q = self.linear_q(x1)  # query
        k = self.linear_k(x2)  # key
        v = self.linear_v(x2)  # value

        # 计算注意力权重
        attn_weights = torch.matmul(q, k.transpose(-2, -1))
        attn_weights = self.softmax(attn_weights)

        # 使用注意力权重加权求和
        attn_output = torch.matmul(attn_weights, v)

        # 输出结果
        output = self.linear_out(attn_output)
        return output


if __name__ == '__main__':
    # 示例输入
    x1 = torch.randn((1, 128, 64, 64))
    x2 = torch.randn((1, 320, 8, 8))

    # 创建交叉注意力模型
    cross_attention = CrossAttention(hidden_dim=1024)

    # 前向传播计算结果
    output = cross_attention(x1, x2)
    print(output)



