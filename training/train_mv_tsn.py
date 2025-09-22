import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import os
import numpy as np
import time
import random

# Import our dataloader
from data.dataloader_coviar import UCF101_TSN_Dataset

# Data augmentation classes
class GroupMultiScaleCrop:
    def __init__(self, input_size, scales):
        self.input_size = input_size
        self.scales = scales if scales is not None else [1, .875, .75, .66]
    def __call__(self, img_group):
        import cv2
        scale = random.choice(self.scales)
        new_size = int(round(self.input_size / scale))
        transformed = []
        for img in img_group:
            img_resized = cv2.resize(img, (new_size, new_size))
            y1, x1 = (random.randint(0, new_size - self.input_size), random.randint(0, new_size - self.input_size))
            transformed.append(img_resized[y1:y1+self.input_size, x1:x1+self.input_size, :])
        return transformed

class GroupRandomHorizontalFlip:
    def __init__(self, is_mv=False):
        self.is_mv = is_mv
    def __call__(self, img_group):
        if random.random() < 0.5:
            flipped = []
            for img in img_group:
                img = np.flip(img, axis=1).copy()
                if self.is_mv: img[:, :, 0] = 255 - img[:, :, 0]
                flipped.append(img)
            return flipped
        return img_group

class GroupScale:
    def __init__(self, new_size): self.new_size = new_size
    def __call__(self, img_group):
        import cv2
        return [cv2.resize(img, (self.new_size, self.new_size)) for img in img_group]

class GroupCenterCrop:
    def __init__(self, crop_size): self.crop_size = crop_size
    def __call__(self, img_group):
        import cv2
        cropped = []
        for img in img_group:
            h, w, _ = img.shape
            y1, x1 = (int(round((h - self.crop_size) / 2)), int(round((w - self.crop_size) / 2)))
            cropped.append(img[y1:y1+self.crop_size, x1:x1+self.crop_size, :])
        return cropped

# MV-TSN Model based on EfficientNet-B0
class MV_TSN_Model(nn.Module):
    def __init__(self, num_classes, base_model='efficientnet_b0'):
        super().__init__()
        print(f"Initializing base model: {base_model} (pretrained)...")
        # Use timm library for easier model loading and modification
        import timm
        self.base_model = timm.create_model(base_model, pretrained=True)
        
        print("Adapting for 2-channel input...")
        # Add a batch normalization layer
        self.data_bn = nn.BatchNorm2d(2)
        
        # Modify EfficientNet's first convolutional layer (conv_stem)
        # Get original weights and parameters
        orig_conv = self.base_model.conv_stem
        orig_weight = orig_conv.weight.clone()
        
        # Create new 2-channel convolutional layer
        self.base_model.conv_stem = nn.Conv2d(2, orig_conv.out_channels, 
                                              kernel_size=orig_conv.kernel_size,
                                              stride=orig_conv.stride,
                                              padding=orig_conv.padding,
                                              bias=False)
        # 将原3通道权重的均值赋给新层
        with torch.no_grad():
            self.base_model.conv_stem.weight.copy_(orig_weight.mean(dim=1, keepdim=True).repeat(1, 2, 1, 1))

        # 替换分类头 (classifier)
        num_ftrs = self.base_model.classifier.in_features
        self.base_model.classifier = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        x = self.data_bn(x)
        return self.base_model(x)

# Learning rate decay strategy
def adjust_learning_rate(optimizer, epoch, lr_steps, lr_decay, base_lr):
    decay = lr_decay ** (sum(epoch >= np.array(lr_steps)))
    lr = base_lr * decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
    return lr

# Training and validation loop
def run_training_loop(model, train_loader, val_loader, criterion, optimizer, num_epochs, lr_params, device):
    best_accuracy = 0.0
    for epoch in range(num_epochs):
        current_lr = adjust_learning_rate(optimizer, epoch, **lr_params)
        print(f"\n--- Epoch {epoch+1}/{num_epochs} | LR: {current_lr:.7f} ---")
        # Training
        model.train()
        running_loss, correct_preds, total_samples = 0.0, 0, 0
        for inputs, labels in tqdm(train_loader, desc="Training"):
            b, n_seg, c, h, w = inputs.shape; inputs = inputs.view(-1, c, h, w).to(device); labels = labels.to(device)
            optimizer.zero_grad(); outputs = model(inputs).view(b, n_seg, -1).mean(dim=1); loss = criterion(outputs, labels)
            loss.backward(); optimizer.step(); running_loss += loss.item() * b; _, preds = torch.max(outputs, 1)
            correct_preds += torch.sum(preds == labels.data); total_samples += b
        train_loss = running_loss / total_samples; train_acc = correct_preds.double() / total_samples
        # Validation
        model.eval(); running_loss, correct_preds, total_samples = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validating"):
                b, n_seg, c, h, w = inputs.shape; inputs = inputs.view(-1, c, h, w).to(device); labels = labels.to(device)
                outputs = model(inputs).view(b, n_seg, -1).mean(dim=1); loss = criterion(outputs, labels)
                running_loss += loss.item() * b; _, preds = torch.max(outputs, 1)
                correct_preds += torch.sum(preds == labels.data); total_samples += b
        val_loss = running_loss / total_samples; val_acc = correct_preds.double() / total_samples
        print(f"Epoch {epoch+1} Summary:\n  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}\n  Valid Loss: {val_loss:.4f}, Valid Acc: {val_acc:.4f}")
        if val_acc > best_accuracy:
            print(f"🚀 New best accuracy! Saving model to 'mv_tsn_efficientnet_best.pth'")
            best_accuracy = val_acc
            torch.save(model.state_dict(), "mv_tsn_efficientnet_best.pth")
    print(f"\n🏆 Best Validation Accuracy: {best_accuracy:.4f}")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    DATA_ROOT = "/home/mbin/data/ucf101/mpeg4_videos"
    TRAIN_LIST_FILE = "/home/mbin/data/datalists/ucf101_split1_train.txt"
    TEST_LIST_FILE  = "/home/mbin/data/datalists/ucf101_split1_test.txt"
    
    # Hyperparameters
    NUM_CLASSES = 101
    NUM_SEGMENTS = 3
    NUM_EPOCHS = 200 # 增加训练周期
    BATCH_SIZE = 80  # 增大Batch Size
    BASE_LR = 0.01   # 更大的初始学习率
    LR_STEPS = [80, 160] # 配合200个epoch的衰减点
    LR_DECAY = 0.1
    WEIGHT_DECAY = 1e-4

    # Switch to EfficientNet-B0 model
    model = MV_TSN_Model(num_classes=NUM_CLASSES, base_model='efficientnet_b0').to(device)

    # Use powerful data augmentation
    train_transform = transforms.Compose([
        GroupMultiScaleCrop(224, scales=[1, 0.875, 0.75]),
        GroupRandomHorizontalFlip(is_mv=True)
    ])
    val_transform = transforms.Compose([
        GroupScale(256),
        GroupCenterCrop(224)
    ])
    print("Loading data with advanced augmentations...")
    train_dataset = UCF101_TSN_Dataset(DATA_ROOT, TRAIN_LIST_FILE, NUM_SEGMENTS, transform=train_transform, is_train=True)
    val_dataset = UCF101_TSN_Dataset(DATA_ROOT, TEST_LIST_FILE, NUM_SEGMENTS, transform=val_transform, is_train=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    # Set differential learning rates
    params_dict = dict(model.named_parameters())
    params = []
    for key, value in params_dict.items():
        decay_mult = 0.0 if 'bias' in key or 'bn' in key else 1.0
        # Set weight decay to 0 for BN layers
        
        # Final classifier layer
        if '.classifier.' in key:
            lr_mult = 1.0
        # Input layer and BN
        elif 'conv_stem' in key or 'data_bn' in key:
            lr_mult = 0.1
        # Other backbone layers
        else:
            lr_mult = 0.01
            
        params.append({'params': value, 'lr_mult': lr_mult, 'decay_mult': decay_mult})

    optimizer = optim.Adam(params, lr=BASE_LR, weight_decay=WEIGHT_DECAY, eps=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Start training
    lr_params = {'lr_steps': LR_STEPS, 'lr_decay': LR_DECAY, 'base_lr': BASE_LR}
    run_training_loop(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, lr_params, device)

if __name__ == '__main__':
    main()