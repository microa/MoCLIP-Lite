import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import cv2
import random

def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

class UCF101Dataset(Dataset):
    def __init__(self, root_video_dir, root_mv_dir, split_file, transform=None, mode='zeroshot'):
        self.root_video_dir = root_video_dir
        self.root_mv_dir = root_mv_dir
        self.split_file = split_file
        self.transform = transform
        self.mode = mode
        
        self.samples = []
        self.class_to_idx = {}

        class_dirs = sorted(os.listdir(self.root_video_dir))
        self.class_to_idx = {class_name: i for i, class_name in enumerate(class_dirs)}
        self.idx_to_class = {i: class_name for class_name, i in self.class_to_idx.items()}

        print(f"Scanning samples for '{os.path.basename(split_file)}'...")
        with open(self.split_file, 'r') as f:
            all_videos = f.readlines()
            for line in all_videos:
                video_rel_path = line.strip().split()[0]
                video_rel_path = video_rel_path.replace('.avi', '')
                class_name = video_rel_path.split('/')[0]
                video_name = os.path.basename(video_rel_path)
                
                video_path = os.path.join(self.root_video_dir, class_name, f"{video_name}.mp4")
                mv_dir_path = os.path.join(self.root_mv_dir, class_name, video_name, "mv_vis")
                
                # 严格检查 mvs-1.png 是否存在
                expected_mv_path = os.path.join(mv_dir_path, "mvs-1.png")
                
                if os.path.exists(video_path) and os.path.exists(expected_mv_path):
                    self.samples.append({
                        "video_path": video_path,
                        "mv_dir_path": mv_dir_path,
                        "class_name": class_name,
                        "label_index": self.class_to_idx[class_name]
                    })
        
        print(f"✅ Dataset for '{os.path.basename(split_file)}' loaded.")
        print(f"   - Found {len(self.samples)} valid samples out of {len(all_videos)} total entries.")
        print(f"   - Mode: '{self.mode}'")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        video_path = sample_info["video_path"]
        label = sample_info["label_index"]

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened(): raise IOError("Cannot open video")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_idx = random.randint(0, total_frames - 1) if total_frames > 0 else 0
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret: raise RuntimeError(f"Failed to read frame {frame_idx}")
            
            cap.release()

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)

            if self.transform:
                frame_pil = self.transform(frame_pil)

            if self.mode == 'zeroshot':
                return frame_pil, label
            
            elif self.mode == 'fusion':
                # 修正索引：MV图编号 = 视频帧索引 + 1
                mv_idx = frame_idx + 1
                mv_img_path = os.path.join(sample_info["mv_dir_path"], f"mvs-{mv_idx}.png")
                
                if not os.path.exists(mv_img_path):
                     mv_img_path = os.path.join(sample_info["mv_dir_path"], "mvs-1.png")

                mv_image = Image.open(mv_img_path).convert("RGB")
                if self.transform:
                    mv_image = self.transform(mv_image)

                return frame_pil, mv_image, label

        except Exception as e:
            print(f"Warning: Runtime error at index {idx} ({video_path}): {e}. Skipping sample.")
            return None