import os
import json
import torch
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm

def get_class_names(root_dir):
    """Gets the sorted list of class names from the directory structure."""
    return sorted(os.listdir(root_dir))

def generate_prompts_for_class(class_name, templates, mappings):
    """Generates all possible prompts for a single class."""
    class_equivalents = mappings.get(class_name, [class_name.lower()])
    final_prompts = []
    for equivalent in class_equivalents:
        for template in templates:
            final_prompts.append(template.format(equivalent))
    return final_prompts

def main():
    # --- 1. Configuration ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ROOT_VIDEO_DIR = "/home/mbin/data/ucf101/mpeg4_videos"
    PROMPT_TEMPLATES_FILE = 'prompt_templates.json'
    CLASS_MAPPINGS_FILE = 'class_mappings.json'
    OUTPUT_FILE = "ucf101_zeroshot_text_features.pt"

    print(f"Using device: {device}")
    
    # --- 2. Load Models and Configs ---
    print("Loading CLIP model and processor...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    print(f"Loading prompts from '{PROMPT_TEMPLATES_FILE}'...")
    with open(PROMPT_TEMPLATES_FILE, 'r') as f:
        templates = json.load(f)

    print(f"Loading class mappings from '{CLASS_MAPPINGS_FILE}'...")
    with open(CLASS_MAPPINGS_FILE, 'r') as f:
        mappings = json.load(f)
        
    class_names = get_class_names(ROOT_VIDEO_DIR)
    print(f"Found {len(class_names)} classes.")

    # --- 3. Generate and Save Features ---
    all_class_features = []
    print("Generating text features for all classes...")
    with torch.no_grad():
        for class_name in tqdm(class_names, desc="Processing classes"):
            # a. Generate all prompt variations for the class
            prompts = generate_prompts_for_class(class_name, templates, mappings)
            
            # b. Get embeddings for all prompts
            inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
            text_features = model.get_text_features(**inputs)
            
            # c. Average the embeddings and normalize
            class_feature = text_features.mean(dim=0)
            class_feature /= class_feature.norm()
            all_class_features.append(class_feature)
    
    # --- 4. Save to File ---
    final_text_features = torch.stack(all_class_features, dim=0)
    torch.save(final_text_features, OUTPUT_FILE)
    
    print(f"\n✅ Successfully generated and saved text features to '{OUTPUT_FILE}'")
    print(f"   - Tensor shape: {final_text_features.shape}")

if __name__ == "__main__":
    main()