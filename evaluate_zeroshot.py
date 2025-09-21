import torch
from transformers import CLIPModel, CLIPProcessor
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np # Still need numpy for final accuracy calculation

# Import our validated Dataloader class
from dataloader import UCF101Dataset 

def run_zeroshot_evaluation():
    # 1. Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ROOT_VIDEO_DIR = "/home/mbin/data/ucf101/mpeg4_videos"
    ROOT_MV_DIR = "/home/mbin/data/ucf101/extract_mvs"
    TEST_SPLIT_FILE  = "/home/mbin/data/datalists/ucf101_split1_test.txt"
    TEXT_FEATURES_FILE = "ucf101_zeroshot_text_features.pt" 
    
    # 2. Load model and data
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    text_features = torch.load(TEXT_FEATURES_FILE, weights_only=True).to(device)
    
    from torchvision import transforms
    preprocess = transforms.Compose([
        transforms.Resize(224), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    test_dataset = UCF101Dataset(
        root_video_dir=ROOT_VIDEO_DIR, root_mv_dir=ROOT_MV_DIR,
        split_file=TEST_SPLIT_FILE, transform=preprocess, mode='zeroshot'
    )
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4) # 加大batch size提高效率

    # Evaluation loop
    model.eval()
    all_predictions = []
    all_labels = []

    print("Starting Zero-Shot evaluation to generate prediction file...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating CLIP"):
            images = images.to(device)
            image_features = model.get_image_features(pixel_values=images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T)
            _, predictions = similarity.max(dim=1)
            
            # Collect PyTorch tensors instead of numpy arrays
            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())

    # Calculate accuracy and save results
    # Merge list of tensors into one large tensor
    final_predictions = torch.cat(all_predictions)
    final_labels = torch.cat(all_labels)

    accuracy = (final_predictions == final_labels).float().mean().item() * 100
    print(f"\n🚀 Final Zero-Shot Top-1 Accuracy: {accuracy:.2f}%")
    
    # Save PyTorch tensors
    print("\nSaving prediction results (as PyTorch Tensors) to 'clip_zeroshot_preds.pt'...")
    torch.save({'preds': final_predictions, 'labels': final_labels}, 'clip_zeroshot_preds.pt')
    print("✅ Done.")

if __name__ == '__main__':
    run_zeroshot_evaluation()