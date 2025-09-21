import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import timm
from tqdm import tqdm
import os

# Import collate_fn
from dataloader import UCF101Dataset, collate_fn

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch的函数"""
    model.train()
    running_loss = 0.0
    # 使用 a temp variable 来记录有效的样本数量
    num_valid_samples = 0
    for batch in tqdm(train_loader, desc="Training"):
        # 因为collate_fn，如果一个批次所有样本都无效，batch可能为None
        if batch is None:
            continue
        
        video_frames, mv_images, labels = batch
        mv_images = mv_images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(mv_images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * mv_images.size(0)
        num_valid_samples += mv_images.size(0)
    
    epoch_loss = running_loss / num_valid_samples if num_valid_samples > 0 else 0
    return epoch_loss

def validate(model, val_loader, criterion, device):
    """验证模型性能的函数"""
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            if batch is None:
                continue
            
            video_frames, mv_images, labels = batch
            mv_images = mv_images.to(device)
            labels = labels.to(device)

            outputs = model(mv_images)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels.data)
            total_samples += labels.size(0)
            running_loss += loss.item() * mv_images.size(0)

    epoch_loss = running_loss / total_samples if total_samples > 0 else 0
    epoch_acc = correct_predictions.double() / total_samples if total_samples > 0 else 0
    return epoch_loss, epoch_acc.item()


def main():
    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    ROOT_VIDEO_DIR = "/home/mbin/data/ucf101/mpeg4_videos"
    ROOT_MV_DIR = "/home/mbin/data/ucf101/extract_mvs"
    TRAIN_SPLIT_FILE = "/home/mbin/data/datalists/ucf101_split1_train.txt"
    TEST_SPLIT_FILE  = "/home/mbin/data/datalists/ucf101_split1_test.txt"
    NUM_EPOCHS = 20
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    NUM_CLASSES = 101
    
    model = timm.create_model('resnet18', pretrained=True, num_classes=NUM_CLASSES)
    model.to(device)

    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print("Loading training data...")
    train_dataset = UCF101Dataset(
        root_video_dir=ROOT_VIDEO_DIR, root_mv_dir=ROOT_MV_DIR,
        split_file=TRAIN_SPLIT_FILE, transform=preprocess, mode='fusion'
    )
    # --- 2. 修改点：在这里增加 collate_fn 参数 ---
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_fn)
    
    print("Loading validation data...")
    val_dataset = UCF101Dataset(
        root_video_dir=ROOT_VIDEO_DIR, root_mv_dir=ROOT_MV_DIR,
        split_file=TEST_SPLIT_FILE, transform=preprocess, mode='fusion'
    )
    # --- 2. 修改点：在这里增加 collate_fn 参数 ---
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    best_accuracy = 0.0
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Validation Loss: {val_loss:.4f}")
        print(f"  Validation Accuracy: {val_acc:.4f}")
        
        if val_acc > best_accuracy:
            print(f"🚀 New best accuracy! Saving model to 'mv_only_best_model.pth'")
            best_accuracy = val_acc
            torch.save(model.state_dict(), "mv_only_best_model.pth")

    print("\n--- Training Finished ---")
    print(f"🏆 Best Validation Accuracy: {best_accuracy:.4f}")

if __name__ == '__main__':
    main()