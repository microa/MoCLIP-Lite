import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np

# 从模块化文件中导入
from dataloader_coviar import UCF101_TSN_Dataset
from model import MV_TSN_Model
from transforms_video import GroupOverSample

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    DATA_ROOT = "/home/mbin/data/ucf101/mpeg4_videos"
    TEST_LIST_FILE  = "/home/mbin/data/datalists/ucf101_split1_test.txt"
    BEST_MODEL_PATH = "mv_tsn_efficientnet_best.pth"
    NUM_CLASSES = 101; NUM_SEGMENTS = 25; BATCH_SIZE = 1; BASE_MODEL = 'efficientnet_b0'

    print(f"Loading best model from '{BEST_MODEL_PATH}'...")
    model = MV_TSN_Model(num_classes=NUM_CLASSES, base_model=BASE_MODEL).to(device)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, weights_only=True))
    model.eval()

    # --- 1. 定义测试变换 ---
    test_transform = GroupOverSample(crop_size=224, scale_size=256, is_mv=True)
    
    # --- 2. 创建Dataloader，并把变换交给他 ---
    print("Loading test data...")
    test_dataset = UCF101_TSN_Dataset(DATA_ROOT, TEST_LIST_FILE, num_segments=NUM_SEGMENTS, 
                                     transform=test_transform, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # --- 3. 简洁的测试循环 ---
    print("Starting intensive testing (25 segments, 10-crop)...")
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            # Dataloader返回的数据已经是正确的5D形状: (B=1, N_crop, N_seg, C, H, W)
            b, n_crop, n_seg, c, h, w = inputs.shape
            
            # 合并crops和segments维度成一个大的batch
            inputs = inputs.view(-1, c, h, w).to(device)
            
            outputs = model(inputs) # -> (N_crop * N_seg, num_classes)
            
            # reshape回来并对所有crops和segments的结果取平均
            outputs = outputs.view(b, n_crop * n_seg, -1).mean(dim=1) # -> (B, num_classes)
            
            _, final_pred = torch.max(outputs, 1)
            
            all_preds.append(final_pred.cpu().item())
            all_labels.append(labels.item())

    accuracy = np.mean(np.array(all_preds) == np.array(all_labels)) * 100
    print(f"\n🏆 Final Testing Accuracy (25-seg, 10-crop): {accuracy:.2f}%")

if __name__ == '__main__':
    main()