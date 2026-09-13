from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch import nn
from .branch import MOC_Branch
from .dla import MOC_DLA
from .resnet import MOC_ResNet

backbone = {
    'dla': MOC_DLA,
    'resnet': MOC_ResNet,
}


class MOC_Net(nn.Module):
    def __init__(self, arch, num_layers, branch_info, head_conv, K, flip_test=False):
        super(MOC_Net, self).__init__()
        self.flip_test = flip_test
        self.K = K
        self.arch = arch
        self.backbone = backbone[arch](num_layers)
        self.branch = MOC_Branch(self.backbone.output_channel, arch, head_conv, branch_info, K)

    def _forward_backbone(self, image, text=None):
        if text is None:
            return self.backbone(image), None
        if self.arch != 'dla':
            raise NotImplementedError('Text-conditioned HCBS fusion is currently implemented only for the DLA backbone.')
        image_feature, text_feature = self.backbone(image, text)
        return image_feature, text_feature

    def forward(self, input, textdata=None):
        if textdata is not None and len(textdata) != len(input):
            raise ValueError('input and textdata must have the same temporal length')

        if self.flip_test:
            assert self.K == len(input) // 2
            chunk1, text1 = [], []
            chunk2, text2 = [], []
            for i in range(self.K):
                t1 = None if textdata is None else textdata[i]
                t2 = None if textdata is None else textdata[i + self.K]
                image_feature, text_feature = self._forward_backbone(input[i], t1)
                chunk1.append(image_feature)
                if text_feature is not None:
                    text1.append(text_feature)

                image_feature, text_feature = self._forward_backbone(input[i + self.K], t2)
                chunk2.append(image_feature)
                if text_feature is not None:
                    text2.append(text_feature)

            return [
                self.branch(chunk1, text1 if text1 else None),
                self.branch(chunk2, text2 if text2 else None),
            ]

        chunk_list = []
        text_data_list = []
        for i in range(self.K):
            text = None if textdata is None else textdata[i]
            image_feature, text_feature = self._forward_backbone(input[i], text)
            chunk_list.append(image_feature)
            if text_feature is not None:
                text_data_list.append(text_feature)

        return [self.branch(chunk_list, text_data_list if text_data_list else None)]
