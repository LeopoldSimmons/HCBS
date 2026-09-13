from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d) and m.bias is not None:
            nn.init.constant_(m.bias, 0)


class GatingModule(nn.Module):
    def __init__(self):
        super(GatingModule, self).__init__()
        self.conv1 = nn.Conv2d(64 * 2, 64, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, image_feature, text_feature):
        text_feature = text_feature / 10
        combined_features = torch.cat((image_feature, text_feature), dim=1)
        gate_weight = self.conv1(combined_features)
        gate_weight = self.conv2(gate_weight)
        gate_weight = self.conv3(gate_weight)
        gate_weight = self.sigmoid(gate_weight)
        return image_feature * (1 - gate_weight) + text_feature * gate_weight


class MOC_Branch(nn.Module):
    def __init__(self, input_channel, arch, head_conv, branch_info, K):
        super(MOC_Branch, self).__init__()
        assert head_conv > 0
        wh_head_conv = 64 if arch == 'resnet' else head_conv
        self.GatingModule = GatingModule()

        self.hm = nn.Sequential(
            nn.Conv2d(K * input_channel, head_conv,
                      kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, branch_info['hm'],
                      kernel_size=1, stride=1,
                      padding=0, bias=True))
        self.hm[-1].bias.data.fill_(-2.19)

        self.mov = nn.Sequential(
            nn.Conv2d(K * input_channel, head_conv,
                      kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, branch_info['mov'],
                      kernel_size=1, stride=1,
                      padding=0, bias=True))
        fill_fc_weights(self.mov)

        self.wh = nn.Sequential(
            nn.Conv2d(input_channel, wh_head_conv,
                      kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(wh_head_conv, branch_info['wh'] // K,
                      kernel_size=1, stride=1,
                      padding=0, bias=True))
        fill_fc_weights(self.wh)

    def forward(self, input_chunk, textdata=None):
        output = {}
        output_wh = []

        if textdata is None:
            merged_feature = torch.cat(input_chunk, dim=1)
        else:
            if len(input_chunk) != len(textdata):
                raise ValueError('input_chunk and textdata must have the same temporal length')
            merged = [self.GatingModule(image_feature, text_feature)
                      for image_feature, text_feature in zip(input_chunk, textdata)]
            merged_feature = torch.cat(merged, dim=1)

        for feature in input_chunk:
            output_wh.append(self.wh(feature))
        input_chunk_cat = torch.cat(input_chunk, dim=1)
        output_wh = torch.cat(output_wh, dim=1)
        output['hm'] = self.hm(merged_feature)
        output['mov'] = self.mov(input_chunk_cat)
        output['wh'] = output_wh
        return output
