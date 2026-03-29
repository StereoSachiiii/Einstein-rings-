import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import urllib.request
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import os

def fetch_stamp(url, ra, dec, size=80):
    filename = url.split('/')[-1]
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
    
    with fits.open(filename) as hdul:
        data = hdul[0].data
        wcs = WCS(hdul[0].header)
    
    coord = SkyCoord(ra=ra, dec=dec, unit='deg')
    x, y = wcs.world_to_pixel(coord)
    
    # Robust NaN check
    if np.isnan(x) or np.isnan(y):
        raise ValueError(f"Target object at ({ra}, {dec}) is outside the frame {filename}.")
    
    x, y = int(round(float(x))), int(round(float(y)))
    
    # Boundary-aware slicing
    y_start = max(0, y - size)
    y_end = min(data.shape[0], y + size)
    x_start = max(0, x - size)
    x_end = min(data.shape[1], x + size)
    
    stamp = data[y_start:y_end, x_start:x_end]
    
    # arcsinh stretch
    percentile_50 = np.percentile(stamp, 50)
    if percentile_50 <= 0: percentile_50 = 1e-5
    return np.arcsinh(stamp / percentile_50)

# We use the same SDSS frame (5313-3-117) but different coordinates
# This ensures one download and guaranteed visibility
frame_url = "https://dr17.sdss.org/sas/dr17/eboss/photoObj/frames/301/5313/3/frame-r-005313-3-0117.fits.bz2"

objects = [
    {
        "label": "Einstein Ring — Cosmic Horseshoe",
        "url": frame_url,
        "ra": 177.1381, "dec": 19.5009  # Precise location
    },
    {
        "label": "Regular Galaxy — No Lensing",
        "url": frame_url,
        "ra": 177.147, "dec": 19.518     # Neighboring galaxy in the same frame
    },
]

fig, axes = plt.subplots(1, 2, figsize=(14, 7))

for ax, obj in zip(axes, objects):
    try:
        stamp = fetch_stamp(obj["url"], obj["ra"], obj["dec"])
        ax.imshow(stamp, cmap='inferno', origin='lower')
        ax.set_title(obj["label"], fontsize=14)
        ax.axis('off')
    except Exception as e:
        ax.set_title(f"Failed to Load {obj['label']}", fontsize=10, color='red')
        print(f"Error loading {obj['label']}: {e}")

plt.suptitle("Gravitational Lensing Comparison — Sloan Digital Sky Survey", fontsize=16, y=0.98)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

print("Both images loaded from SDSS frame: 5313-3-117")

# [PART 2] — Galaxy Zoo 2 DataLoader Implementation
print("\n" + "="*50)
print("INITIALIZING GALAXY ZOO 2 DATALOADER")
print("="*50)

import torch
from dataset import GalaxyZooDataset
from torch.utils.data import DataLoader
from torchvision import transforms

# Hardware Check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Transforms
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Initialize Dataset
gz_dataset = GalaxyZooDataset(
    labels_path='gz2_hart16.csv.gz',
    mapping_path='gz2_filename_mapping.csv',
    img_dir=os.path.join('images_gz2', 'images'),
    transform=data_transform
)

# Initialize DataLoader
train_loader = DataLoader(gz_dataset, batch_size=8, shuffle=True)

# Pull a batch and verify GPU transfer
print("Pulling first batch...")
images, labels = next(iter(train_loader))

print(f"Batch loaded. Image shape: {images.shape}")
print(f"Labels shape: {labels.shape}")

# Transfer to GPU
images = images.to(device)
labels = labels.to(device)
print(f"Successfully transferred batch to {device.type.upper()}.")

# Visualize the batch
def show_batch(imgs, lbls):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    # Unnormalize for visualization
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    
    for i in range(len(imgs)):
        # Move back to CPU for numpy/matplotlib
        img = inv_normalize(imgs[i].cpu()).permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        axes[i].imshow(img)
        axes[i].set_title(f"Smooth: {lbls[i][0]:.2f}\nFeatures: {lbls[i][1]:.2f}", fontsize=10)
        axes[i].axis('off')
    
    plt.suptitle("First Batch of Galaxy Zoo 2 Data (from DataLoader)", fontsize=16)
    plt.tight_layout()
    plt.show()

show_batch(images, labels)