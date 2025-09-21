import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np

# Import from modular files
from dataloader_coviar import UCF101_Fusion_Dataset # Use new Fusion Dataloader
from train_fusion import LateFusionModel # Import model from fusion training script
from transforms_video import GroupOverSample

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Path configuration
    DATA_ROOT = "/home/mbin/data/ucf101/mpeg4_videos"
    TEST_LIST_FILE  = "/home/mbin/data/datalists/ucf101_split1_test.txt"
    CLIP_FEATURES_DIR = "/home/mbin/data/ucf101/clip_features"
    MV_MODEL_PATH = "mv_tsn_efficientnet_best.pth" # Required for initialization
    BEST_FUSION_MODEL_PATH = "fusion_best_model.pth" # Model to test

    # Test hyperparameters
    NUM_CLASSES = 101
    NUM_SEGMENTS = 25
    BASE_MODEL = 'efficientnet_b0'
    CLIP_FEATURE_DIM = 512
    BATCH_SIZE = 1 # Must be 1

    print(f"Loading best fusion model from '{BEST_FUSION_MODEL_PATH}'...")
    # 1. First instantiate model structure
    model = LateFusionModel(num_classes=NUM_CLASSES, 
                            mv_model_path=MV_MODEL_PATH, 
                            clip_feature_dim=CLIP_FEATURE_DIM).to(device)
    # 2. Load trained fusion model weights
    model.load_state_dict(torch.load(BEST_FUSION_MODEL_PATH, weights_only=True))
    model.eval()

    # Define test transforms
    test_transform = GroupOverSample(crop_size=224, scale_size=256, is_mv=True)
    
    # Create Dataloader
    print("Loading test data for fusion model...")
    test_dataset = UCF101_Fusion_Dataset(
        data_root=DATA_ROOT, video_list_file=TEST_LIST_FILE, 
        num_segments=NUM_SEGMENTS, transform=test_transform, 
        is_train=False, clip_feature_dir=CLIP_FEATURES_DIR
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print("Starting intensive testing on fusion model (25 segments, 10-crop)...")
    all_preds, all_labels = [], []
    with torch.no_grad():
        for mv_data, clip_features, labels in tqdm(test_loader, desc="Testing Fusion Model"):
            # mv_data shape: (B=1, N_crop, N_seg, C, H, W)
            b, n_crop, n_seg, c, h, w = mv_data.shape
            
            # 只有MV数据需要被reshape和通过模型
            mv_data = mv_data.view(-1, c, h, w).to(device)
            clip_features = clip_features.to(device) # CLIP特征已经是(B, 512)
            
            # 使用模型进行前向传播
            outputs = model(mv_data, clip_features) # 模型内部会处理好平均和拼接
            
            _, final_pred = torch.max(outputs, 1)
            
            all_preds.append(final_pred.cpu().item())
            all_labels.append(labels.item())

    accuracy = np.mean(np.array(all_preds) == np.array(all_labels)) * 100
    print(f"\n🏆 Final Fusion Model Testing Accuracy (25-seg, 10-crop): {accuracy:.2f}%")

if __name__ == '__main__':
    main()