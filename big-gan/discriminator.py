import math
import torch.nn as nn
from layers import SNConv2d,DisBlock,SNLinear
import torch.nn.functional as F
import torch

class BigGANDiscriminator(nn.Module):
    def __init__(self, num_classes, ch=64, img_size=128, attention_res=[64, 32]):
        super(BigGANDiscriminator, self).__init__()
        self.num_classes = num_classes
        self.ch = ch
        self.img_size = img_size
        self.attention_res = attention_res

        self.first_conv = SNConv2d(3, ch, kernel_size=3, padding=1, bias=False)
        # Removed inplace=True from LeakyReLU
        self.relu = nn.LeakyReLU(0.2)

        self.blocks = nn.ModuleList()
        current_in_channels = ch
        current_res = img_size

        # Determine number of downsampling steps to reach target_res=4x4
        num_downsample_steps = int(math.log2(img_size / 4))

        for i in range(num_downsample_steps):
            downsample = True
            # Attention is applied when the *output* resolution of the block matches attention_res
            use_attention = (current_res // 2 in self.attention_res)

            # Channels typically double with each downsample, but cap at ch*16
            next_out_channels = min(ch * 16, current_in_channels * 2)

            block = DisBlock(current_in_channels, next_out_channels, downsample=downsample, attention=use_attention)
            self.blocks.append(block)
            current_in_channels = next_out_channels # Update for next iteration
            current_res //= 2

        self.final_features_channels = current_in_channels # Should be ch*16 for 4x4 resolution

        self.linear_out = SNLinear(self.final_features_channels, 1) # Unconditional score
        self.embed_y = SNLinear(num_classes, self.final_features_channels) # Class embedding for projection

    def forward(self, x, y):
        out = self.relu(self.first_conv(x))

        for block in self.blocks:
            out = block(out)

        # Global sum pooling to reduce 4x4 to 1x1
        out = torch.sum(self.relu(out), dim=[2, 3])

        # Unconditional score
        uncond_score = self.linear_out(out)

        # Conditional score (Projection Discriminator)
        y_embed = self.embed_y(F.one_hot(y, num_classes=self.num_classes).float())
        cond_score = torch.sum(out * y_embed, dim=1, keepdim=True)

        return uncond_score + cond_score

