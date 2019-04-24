# -----------------------------
# -*- coding:utf-8 -*-
# author:kangkang
# datetime:2019/4/22 11:42
# -----------------------------

import torch
import torch.nn as nn
import torchvision
from torchsummary import summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module):
    """
    Encoder.
    """
    def __init__(self, basemodel='res101', encoded_image_size=14):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size
        self.basemodel = basemodel
        if self.basemodel == 'vgg16':
            model = torchvision.models.vgg16(pretrained=True)  # pretrained ImageNet ResNet-101
            encoded_image_size = 28
            # Remove linear and pool layers (since we're not doing classification)
            modules = list(model.features)[:-1]
            self.model = nn.Sequential(*modules)
            # self.model输出大小是[-1, 512, 32, 32]
            # Resize image to fixed size to allow input images of variable size
            self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

            self.fine_tune()
        else:
            model = torchvision.models.resnet101(pretrained=True)  # pretrained ImageNet ResNet-101

            # Remove linear and pool layers (since we're not doing classification)
            modules = list(model.children())[:-2]
            self.model = nn.Sequential(*modules)
            # self.model输出大小是[-1, 2048, 16, 16]
            # Resize image to fixed size to allow input images of variable size
            self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

            self.fine_tune()

    def forward(self, images):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.model(images)  # (batch_size, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        return out

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        if self.basemodel == 'vgg16':
            layer = -6
        else:
            layer = -7

        for p in self.model.parameters():
            p.requires_grad = False
            # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.model.children())[layer:]:
            for p in c.parameters():
                p.requires_grad = fine_tune

class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha

encoder = Encoder().to(device)
d = torch.randn(1, 3, 512, 512).cuda()
out = encoder(d)
print(out.shape)
summary(encoder,(3,512,512))