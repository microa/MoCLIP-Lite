import os
import random
import numpy as np
import torch
import torch.utils.data as data
import cv2

try:
    from coviar import get_num_frames, load
except ImportError:
    raise ImportError("Please install `coviar` library first: pip install coviar")

# ... (helper functions clip_and_scale, get_seg_range, get_gop_pos remain the same) ...
def clip_and_scale(img, size): return (img * (127.5 / size)).astype(np.int32)
def get_seg_range(n, num_segments, seg):
    n-=1; seg_size=float(n-1)/num_segments; seg_begin=int(np.round(seg_size*seg)); seg_end=int(np.round(seg_size*(seg+1)))
    if seg_end==seg_begin: seg_end=seg_begin+1
    return seg_begin+1, seg_end+1
def get_gop_pos(frame_idx):
    GOP_SIZE=12; gop_index=frame_idx//GOP_SIZE; gop_pos=frame_idx%GOP_SIZE
    if gop_pos==0: gop_index-=1; gop_pos=GOP_SIZE-1
    return gop_index, gop_pos

class UCF101_TSN_Dataset(data.Dataset):
    def __init__(self, data_root, video_list_file, num_segments, transform, is_train=True):
        self.data_root = data_root
        self.video_list_file = video_list_file
        self.num_segments = num_segments
        self.transform = transform
        self.is_train = is_train
        self.videos = []
        
        with open(video_list_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3: continue
                video_rel_path, _, label = parts[0], parts[1], int(parts[2])
                full_path = os.path.join(self.data_root, video_rel_path.replace('.avi', '.mp4'))
                if os.path.exists(full_path):
                    try:
                        num_frames = get_num_frames(full_path)
                        if num_frames > self.num_segments:
                            self.videos.append((full_path, label, num_frames))
                    except Exception: pass
        print(f"✅ Dataset loaded from '{video_list_file}'. Found {len(self.videos)} valid videos.")

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        video_path, label, num_frames = self.videos[index]
        segments = []
        for seg_idx in range(self.num_segments):
            try:
                if self.is_train:
                    seg_begin, seg_end = get_seg_range(num_frames, self.num_segments, seg_idx)
                    frame_idx = random.randint(seg_begin, seg_end - 1)
                else:
                    seg_size = float(num_frames - 1) / self.num_segments
                    frame_idx = int(np.round(seg_size * (seg_idx + 0.5))) + 1
                gop_index, gop_pos = get_gop_pos(frame_idx)
                img = load(video_path, gop_index, gop_pos, 1, True)
                if img is None: raise ValueError("coviar returned None")
                img = clip_and_scale(img, 20); img += 128; img = np.clip(img, 0, 255).astype(np.uint8)
                segments.append(img)
            except Exception:
                segments.append(np.zeros((256, 256, 2), dtype=np.uint8))

        if self.transform:
            segments = self.transform(segments)
            
        input_tensor = np.array(segments)
        if input_tensor.ndim == 4:
            input_tensor = input_tensor.transpose((0, 3, 1, 2))
        elif input_tensor.ndim == 5:
            input_tensor = input_tensor.transpose((0, 1, 4, 2, 3))
            
        input_tensor = torch.from_numpy(np.ascontiguousarray(input_tensor)).float() / 255.0 - 0.5
        return input_tensor, label

# --- 新增：专门用于融合模型的Dataloader ---
class UCF101_Fusion_Dataset(UCF101_TSN_Dataset):
    def __init__(self, clip_feature_dir, **kwargs):
        super().__init__(**kwargs)
        self.clip_feature_dir = clip_feature_dir

    def __getitem__(self, index):
        # 1. 调用父类方法，获取MV数据和标签
        mv_input_tensor, label = super().__getitem__(index)
        
        # 2. 加载对应的预计算CLIP特征
        video_path, _, _ = self.videos[index]
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        feature_path = os.path.join(self.clip_feature_dir, f"{video_name}.pt")
        
        try:
            clip_feature = torch.load(feature_path, weights_only=True)
        except FileNotFoundError:
            print(f"Warning: CLIP feature not found for {video_name}. Using zeros.")
            clip_feature = torch.zeros(512) # CLIP base model's feature size
            
        return mv_input_tensor, clip_feature, label