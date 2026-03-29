import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class GalaxyZooDataset(Dataset):
    def __init__(self, labels_path, mapping_path, img_dir, transform=None):
        """
        Args:
            labels_path (string): Path to the labels CSV (gz2_hart16.csv.gz).
            mapping_path (string): Path to the mapping CSV (gz2_filename_mapping.csv).
            img_dir (string): Directory with all the images (images_gz2/images).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        print("Loading and merging metadata...")
        labels_df = pd.read_csv(labels_path)
        mapping_df = pd.read_csv(mapping_path)
        
        # Merge on dr7objid (labels) and objid (mapping)
        self.df = pd.merge(
            labels_df, 
            mapping_df[['objid', 'asset_id']], 
            left_on='dr7objid', 
            right_on='objid'
        )
        
        self.img_dir = img_dir
        self.transform = transform
        
        # Define the target columns (Smooth vs Features)
        self.label_cols = [
            't01_smooth_or_features_a01_smooth_fraction',
            't01_smooth_or_features_a02_features_or_disk_fraction'
        ]
        
        print(f"Dataset initialized with {len(self.df)} samples.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        asset_id = self.df.iloc[idx]['asset_id']
        img_name = os.path.join(self.img_dir, f"{asset_id}.jpg")
        
        try:
            image = Image.open(img_name).convert('RGB')
        except FileNotFoundError:
            # Fallback for missing images if any
            print(f"Warning: Image {img_name} not found. Returning a blank image.")
            image = Image.new('RGB', (424, 424), (0, 0, 0))

        labels = self.df.iloc[idx][self.label_cols].values.astype('float32')
        labels = torch.tensor(labels)

        if self.transform:
            image = self.transform(image)

        return image, labels

if __name__ == "__main__":
    # Internal Verification
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = GalaxyZooDataset(
        labels_path='gz2_hart16.csv.gz',
        mapping_path='gz2_filename_mapping.csv',
        img_dir=os.path.join('images_gz2', 'images'),
        transform=data_transform
    )

    img, lbl = dataset[0]
    print(f"Sample image shape: {img.shape}")
    print(f"Sample labels (Smooth, Features): {lbl}")
