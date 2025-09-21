import torch
import torch.nn as nn
import timm

class MV_TSN_Model(nn.Module):
    def __init__(self, num_classes, base_model='efficientnet_b0'):
        super().__init__()
        print(f"Initializing base model: {base_model} (pretrained)...")
        self.base_model = timm.create_model(base_model, pretrained=True)
        
        print("Adapting for 2-channel input...")
        self.data_bn = nn.BatchNorm2d(2)
        
        # Modify EfficientNet's first convolutional layer (conv_stem)
        orig_conv = self.base_model.conv_stem
        orig_weight = orig_conv.weight.clone()
        
        self.base_model.conv_stem = nn.Conv2d(2, orig_conv.out_channels, 
                                              kernel_size=orig_conv.kernel_size,
                                              stride=orig_conv.stride,
                                              padding=orig_conv.padding,
                                              bias=False)
        with torch.no_grad():
            self.base_model.conv_stem.weight.copy_(orig_weight.mean(dim=1, keepdim=True).repeat(1, 2, 1, 1))

        # Replace classifier head
        num_ftrs = self.base_model.classifier.in_features
        self.base_model.classifier = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        # x shape: (B * N_seg * N_crop, C, H, W)
        x = self.data_bn(x)
        return self.base_model(x)