import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import pandas as pd
import numpy as np

def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model, device

def run_validation_v20(model_path, output_file='test_20.txt'):
    model, device = load_model(model_path)
    
    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("Loading metadata...")
    labels_df = pd.read_csv('gz2_hart16.csv.gz')
    mapping_df = pd.read_csv('gz2_filename_mapping.csv')
    
    # Merge for easier sampling
    df = pd.merge(labels_df[['dr7objid', 'rastring', 'decstring', 't01_smooth_or_features_a01_smooth_fraction', 't01_smooth_or_features_a02_features_or_disk_fraction']],
                  mapping_df[['objid', 'asset_id']],
                  left_on='dr7objid', right_on='objid')
    
    # Sample 20 random galaxies
    sample_df = df.sample(500, random_state=42)
    
    results = []
    results.append("="*80)
    results.append("GALAXY ZOO 2 VALIDATION REPORT - 20 RANDOM SAMPLES")
    results.append("="*80)
    results.append("%-10s %-25s %-15s %-15s %-10s" % ("Asset ID", "SDSS Name", "Predicted (S/F)", "Actual (S/F)", "Error (MSE)"))
    results.append("-" * 80)
    
    total_mse = 0
    
    print("Running inference on 20 samples...")
    for idx, row in sample_df.iterrows():
        asset_id = row['asset_id']
        img_path = f"images_gz2/images/{asset_id}.jpg"
        
        if not os.path.exists(img_path):
            continue
            
        # Actual labels
        actual_s = row['t01_smooth_or_features_a01_smooth_fraction']
        actual_f = row['t01_smooth_or_features_a02_features_or_disk_fraction']
        
        # Inference
        img = Image.open(img_path).convert('RGB')
        tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(tensor).cpu().numpy()[0]
        
        # Predicted labels
        pred_s, pred_f = output[0], output[1]
        
        # Calculate Error for this galaxy
        mse = ((pred_s - actual_s)**2 + (pred_f - actual_f)**2) / 2
        total_mse += mse
        
        # Format name
        name = f"J{row['rastring']}{row['decstring']}".replace(":", "").replace(" ", "")
        
        line = "%-10s %-25s %-15s %-15s %-10.4f" % (
            asset_id, name, 
            f"{pred_s:.2f}/{pred_f:.2f}", 
            f"{actual_s:.2f}/{actual_f:.2f}",
            mse
        )
        results.append(line)
    
    avg_mse = total_mse / 20
    results.append("-" * 80)
    results.append(f"AVERAGE MSE ACROSS 20 SAMPLES: {avg_mse:.6f}")
    results.append("="*80)
    
    with open(output_file, 'w') as f:
        f.write("\n".join(results))
    
    print(f"Report generated: {output_file}")

if __name__ == "__main__":
    model_path = 'galaxy_model_best.pth'
    if not os.path.exists(model_path):
        model_path = 'galaxy_model.pth'
    
    run_validation_v20(model_path)
