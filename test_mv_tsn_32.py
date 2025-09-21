import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np

# 从模块化文件中导入
from dataloader_coviar import UCF101_TSN_Dataset
from model import MV_TSN_Model
# Import new class
from transforms_video import GroupCenterSample

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    DATA_ROOT = "/home/mbin/data/ucf101/mpeg4_videos"
    TEST_LIST_FILE  = "/home/mbin/data/datalists/ucf101_split1_test.txt"
    BEST_MODEL_PATH = "mv_tsn_efficientnet_best.pth"
    NUM_CLASSES = 101; NUM_SEGMENTS = 32; BATCH_SIZE = 1; BASE_MODEL = 'efficientnet_b0'

    print(f"Loading best model from '{BEST_MODEL_PATH}'...")
    model = MV_TSN_Model(num_classes=NUM_CLASSES, base_model=BASE_MODEL).to(device)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, weights_only=True))
    model.eval()

    # Define test transforms
    test_transform = GroupCenterSample(crop_size=224, scale_size=256)
    # Create Dataloader
    print("Loading test data...")
    test_dataset = UCF101_TSN_Dataset(DATA_ROOT, TEST_LIST_FILE, num_segments=NUM_SEGMENTS, 
                                      transform=test_transform, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Test loop
    print("Starting intensive testing (32 segments, 1-crop)...")
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            # Dataloader返回的数据现在形状是: (B=1, N_crop=1, N_seg, C, H, W)
            b, n_crop, n_seg, c, h, w = inputs.shape
            inputs = inputs.view(-1, c, h, w).to(device)
            outputs = model(inputs)
            outputs = outputs.view(b, n_crop * n_seg, -1).mean(dim=1)
            _, final_pred = torch.max(outputs, 1)
            all_preds.append(final_pred.cpu().item())
            all_labels.append(labels.item())

    accuracy = np.mean(np.array(all_preds) == np.array(all_labels)) * 100
    print(f"\n🏆 Final Testing Accuracy (25-seg, 1-crop): {accuracy:.2f}%")

if __name__ == '__main__':
    main()