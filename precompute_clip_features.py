import torch
from transformers import CLIPModel, CLIPProcessor
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# 导入我们最开始那个只加载原始视频帧的Dataloader
from dataloader import UCF101Dataset 

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- 配置 ---
    ROOT_VIDEO_DIR = "/home/mbin/data/ucf101/mpeg4_videos"
    ROOT_MV_DIR = "/home/mbin/data/ucf101/extract_mvs"
    # 我们需要遍历训练集和测试集
    SPLIT_FILES = [
        "/home/mbin/data/datalists/ucf101_split1_train.txt",
        "/home/mbin/data/datalists/ucf101_split1_test.txt"
    ]
    # 创建一个目录来存放CLIP特征
    CLIP_FEATURES_DIR = "/home/mbin/data/ucf101/clip_features"
    os.makedirs(CLIP_FEATURES_DIR, exist_ok=True)

    # --- 加载模型 ---
    print("Loading CLIP model...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # --- 循环处理每个划分 ---
    for split_file in SPLIT_FILES:
        print(f"\nProcessing split file: {os.path.basename(split_file)}")
        dataset = UCF101Dataset(
            root_video_dir=ROOT_VIDEO_DIR, root_mv_dir=ROOT_MV_DIR,
            split_file=split_file, mode='zeroshot',
            # 使用CLIP官方的预处理器
            transform=transforms.Compose([
                transforms.Resize(224), transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
        )
        
        loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=8)

        model.eval()
        with torch.no_grad():
            for i, (images, labels) in enumerate(tqdm(loader, desc="Extracting features")):
                images = images.to(device)
                
                # 提取特征
                image_features = model.get_image_features(pixel_values=images).cpu()
                
                # 逐一保存每个视频的特征
                for j in range(image_features.shape[0]):
                    sample_idx = i * loader.batch_size + j
                    if sample_idx < len(dataset.samples):
                        video_path = dataset.samples[sample_idx]["video_path"]
                        video_name = os.path.splitext(os.path.basename(video_path))[0]
                        feature_path = os.path.join(CLIP_FEATURES_DIR, f"{video_name}.pt")
                        torch.save(image_features[j], feature_path)

    print("\n✅ All CLIP features have been pre-computed and saved.")

if __name__ == '__main__':
    from torchvision import transforms
    main()