import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np

# Core imports
from torchvision import transforms
from data.dataloader_coviar import UCF101_TSN_Dataset
from models.model import MV_TSN_Model 
from data.transforms_video import GroupMultiScaleCrop, GroupRandomHorizontalFlip, GroupScale, GroupCenterCrop

class LateFusionModel(nn.Module):
    def __init__(self, num_classes, mv_model_path, clip_feature_dim=512):
        super().__init__()
        self.mv_feature_extractor = MV_TSN_Model(num_classes=num_classes)
        print(f"Loading pretrained MV model weights from {mv_model_path}")
        self.mv_feature_extractor.load_state_dict(torch.load(mv_model_path, weights_only=True))
        
        mv_feature_dim = self.mv_feature_extractor.base_model.classifier.in_features
        self.mv_feature_extractor.base_model.classifier = nn.Identity()

        self.fusion_classifier = nn.Sequential(
            nn.Linear(mv_feature_dim + clip_feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, mv_data, clip_features):
        b = clip_features.shape[0]
        n_seg = mv_data.shape[0] // b
        
        mv_features = self.mv_feature_extractor(mv_data)
        mv_features = mv_features.view(b, n_seg, -1).mean(dim=1)

        combined_features = torch.cat([mv_features, clip_features], dim=1)
        
        output = self.fusion_classifier(combined_features)
        return output

class UCF101_Fusion_Dataset(UCF101_TSN_Dataset):
    def __init__(self, clip_feature_dir, **kwargs):
        super().__init__(**kwargs)
        self.clip_feature_dir = clip_feature_dir

    def __getitem__(self, index):
        mv_input_tensor, label = super().__getitem__(index)
        
        video_path, _, _ = self.videos[index]
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        feature_path = os.path.join(self.clip_feature_dir, f"{video_name}.pt")
        
        try:
            clip_feature = torch.load(feature_path, weights_only=True)
        except FileNotFoundError:
            print(f"Warning: CLIP feature not found for {video_name}. Using zeros.")
            clip_feature = torch.zeros(512)
            
        return mv_input_tensor, clip_feature, label

def run_fusion_training(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    best_accuracy = 0.0
    for epoch in range(num_epochs):
        print(f"\n--- Fusion Epoch {epoch+1}/{num_epochs} ---")
        model.train()
        running_loss, correct_preds, total_samples = 0.0, 0, 0
        for mv_data, clip_features, labels in tqdm(train_loader, desc="Fusion Training"):
            # Handle potential None batches from dataloader
            if mv_data is None: continue
            
            b, n_seg, c, h, w = mv_data.shape
            mv_data = mv_data.view(-1, c, h, w).to(device)
            clip_features = clip_features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(mv_data, clip_features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            correct_preds += torch.sum(preds == labels.data)
            total_samples += b
            running_loss += loss.item() * b
        
        train_acc = (correct_preds.double() / total_samples) if total_samples > 0 else 0
        
        model.eval()
        correct_preds, total_samples = 0, 0
        with torch.no_grad():
            for mv_data, clip_features, labels in tqdm(val_loader, desc="Fusion Validating"):
                if mv_data is None: continue

                b, n_seg, c, h, w = mv_data.shape
                mv_data = mv_data.view(-1, c, h, w).to(device)
                clip_features = clip_features.to(device)
                labels = labels.to(device)
                outputs = model(mv_data, clip_features)
                _, preds = torch.max(outputs, 1)
                correct_preds += torch.sum(preds == labels.data)
                total_samples += b
        
        val_acc = (correct_preds.double() / total_samples) if total_samples > 0 else 0
        print(f"Epoch {epoch+1} Summary: Train Acc: {train_acc:.4f}, Valid Acc: {val_acc:.4f}")

        if val_acc > best_accuracy:
            print(f"🚀 New best fusion accuracy! Saving model to 'fusion_best_model.pth'")
            best_accuracy = val_acc
            torch.save(model.state_dict(), "fusion_best_model.pth")
    print(f"\n🏆 Best Fusion Validation Accuracy: {best_accuracy:.4f}")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    DATA_ROOT = "/home/mbin/data/ucf101/mpeg4_videos"
    TRAIN_LIST_FILE = "/home/mbin/data/datalists/ucf101_split1_train.txt"
    TEST_LIST_FILE  = "/home/mbin/data/datalists/ucf101_split1_test.txt"
    CLIP_FEATURES_DIR = "/home/mbin/data/ucf101/clip_features"
    MV_MODEL_PATH = "mv_tsn_efficientnet_best.pth"

    NUM_CLASSES = 101; NUM_SEGMENTS = 3; NUM_EPOCHS = 100; BATCH_SIZE = 64; LEARNING_RATE = 1e-4

    model = LateFusionModel(num_classes=NUM_CLASSES, mv_model_path=MV_MODEL_PATH).to(device)

    train_transform = transforms.Compose([
        GroupMultiScaleCrop(224, scales=[1, 0.875, 0.75]), GroupRandomHorizontalFlip(is_mv=True)
    ])
    val_transform = transforms.Compose([
        GroupScale(256), GroupCenterCrop(224)
    ])
    
    train_dataset = UCF101_Fusion_Dataset(
        data_root=DATA_ROOT, video_list_file=TRAIN_LIST_FILE, num_segments=NUM_SEGMENTS,
        transform=train_transform, is_train=True, clip_feature_dir=CLIP_FEATURES_DIR
    )
    val_dataset = UCF101_Fusion_Dataset(
        data_root=DATA_ROOT, video_list_file=TEST_LIST_FILE, num_segments=NUM_SEGMENTS,
        transform=val_transform, is_train=False, clip_feature_dir=CLIP_FEATURES_DIR
    )
    # Using a simple collate_fn to handle potential None from dataloader
    def collate_fn_skip_none(batch):
        batch = [item for item in batch if item is not None]
        if not batch: return None, None, None
        return torch.utils.data.dataloader.default_collate(batch)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, collate_fn=collate_fn_skip_none)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, collate_fn=collate_fn_skip_none)

    # We only train the new MLP classifier
    optimizer = optim.AdamW(model.fusion_classifier.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    run_fusion_training(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, device)

if __name__ == '__main__':
    main()