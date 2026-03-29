import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import pandas as pd
import matplotlib.pyplot as plt

def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model to: {device}")
    
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Weights loaded from {model_path}")
    else:
        raise FileNotFoundError(f"No model found at {model_path}. Please train the model first.")
    
    model = model.to(device)
    model.eval()
    return model, device

def get_galaxy_metadata(asset_id, labels_path='gz2_hart16.csv.gz', mapping_path='gz2_filename_mapping.csv'):
    """Looks up the actual Hart et al. (2016) labels and name for a given asset_id."""
    print("Loading metadata for verification...")
    labels_df = pd.read_csv(labels_path)
    mapping_df = pd.read_csv(mapping_path)
    
    # Bridge asset_id -> objid -> labels
    objid = mapping_df[mapping_df['asset_id'] == int(asset_id)]['objid'].values[0]
    row = labels_df[labels_df['dr7objid'] == objid]
    
    actual_smooth = row['t01_smooth_or_features_a01_smooth_fraction'].values[0]
    actual_features = row['t01_smooth_or_features_a02_features_or_disk_fraction'].values[0]
    
    # Extract celestial name (RA/Dec string)
    ra_str = row['rastring'].values[0]
    dec_str = row['decstring'].values[0]
    # Standard SDSS format: J{RA}{Dec} - remove colons for naming convention
    ra_clean = ra_str.replace(":", "")
    dec_clean = dec_str.replace(":", "")
    obj_name = f"SDSS J{ra_clean}{dec_clean}".replace(" ", "")
    
    return obj_name, [actual_smooth, actual_features]

def predict_galaxy(model, device, image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        predictions = output.cpu().numpy()[0]
    
    return predictions

def visualize_prediction(image_path, obj_name, pred, actual=None):
    image = Image.open(image_path)
    fig = plt.figure(figsize=(10, 8))
    
    # Explicitly set the OS window title
    try:
        fig.canvas.manager.set_window_title(f"Galaxy Zoo 2 - {obj_name}")
    except:
        pass # Some backends might not support this
        
    plt.imshow(image)
    plt.axis('off')
    
    # Massive title for maximum visibility
    title = f"OBJECT NAME: {obj_name}\n"
    title += "="*40 + "\n"
    title += f"MODEL PREDICTION:\nSmooth: {pred[0]:.4f} | Features: {pred[1]:.4f}"
    if actual is not None:
        title += f"\n\nGROUND TRUTH (Expert Label):\nSmooth: {actual[0]:.4f} | Features: {actual[1]:.4f}"
    
    plt.title(title, fontsize=14, fontweight='bold', color='darkblue')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage
    asset_id = "138031" # You can change this to any valid asset_id
    img_path = f"images_gz2/images/{asset_id}.jpg"
    
    model_file = "galaxy_model_best.pth"
    if not os.path.exists(model_file):
         model_file = "galaxy_model.pth"

    try:
        model, device = load_model(model_file)
        
        # 1. Fetch Metadata (Name and Ground Truth)
        obj_name, actual_labels = get_galaxy_metadata(asset_id)
        
        # 2. Run Prediction
        pred_probs = predict_galaxy(model, device, img_path)
        
        print("\n" + "#"*60)
        print(f"### GALAXY IDENTIFIED: {obj_name}")
        print("#"*60)
        print(f"ID (Asset/Obj): {asset_id} / {obj_name}")
        print("-" * 60)
        print(f"%-20s %-15s %-15s" % ("Morphology", "Prediction", "Expert (GZ2)"))
        print("-" * 60)
        print(f"%-20s %-15.4f %-15.4f" % ("Smooth", pred_probs[0], actual_labels[0]))
        print(f"%-20s %-15.4f %-15.4f" % ("Features/Disk", pred_probs[1], actual_labels[1]))
        print("#"*60 + "\n")
        
        visualize_prediction(img_path, obj_name, pred_probs, actual_labels)
        
    except Exception as e:
        print(f"Error: {e}")
