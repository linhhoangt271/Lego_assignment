"""
TensorBoard Visualization for V3 Augmentations & Training Data
Shows: Original images, augmented samples, class distributions, augmentation effects
"""

import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageOps
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_SIZE = 380
BATCH_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_DIR = os.path.join(BASE_DIR, "runs", "v3_augmentation_viz")
os.makedirs(LOG_DIR, exist_ok=True)

print(f"Using device: {DEVICE}")
print(f"TensorBoard logs: {LOG_DIR}")

# ============================================================
# LOAD METADATA
# ============================================================
with open(os.path.join(BASE_DIR, "minifigs.json")) as f:
    minifigs_list = json.load(f)

valid_images = set()
theme_map = {}
theme_path_map = {}

for minifig in minifigs_list:
    img_path = minifig.get("img_local_path")
    theme = minifig.get("themes", [None])[0]

    if img_path and theme and os.path.exists(img_path):
        valid_images.add(img_path)
        theme_map[img_path] = theme
        theme_path_map[img_path] = img_path

print(f"Loaded {len(valid_images)} valid images from {len(set(theme_map.values()))} themes")

# ============================================================
# MERGE & SPLIT LOGIC (from V3)
# ============================================================
MERGE_MAP = {
    'The LEGO NINJAGO Movie': 'NINJAGO',
    'The LEGO Movie': 'LEGO Movies', 'The LEGO Movie 2': 'LEGO Movies',
    'DC Super Hero Girls': 'Super Heroes', 'Spider-Man': 'Super Heroes', 'Batman I': 'Super Heroes',
    'Elves': 'Friends & Fantasy', 'Friends': 'Friends & Fantasy',
    'DUPLO': 'Preschool', 'Primo': 'Preschool', 'Belville': 'Preschool', 'Scala': 'Preschool',
    'For Juniors': 'Preschool', 'Homemaker': 'Preschool', 'Fabuland': 'Preschool',
    'Pirates of the Caribbean': 'Pirates',
    'Castle': 'Castle & Medieval', 'Vikings': 'Castle & Medieval', 'Ninja': 'Castle & Medieval',
    'NEXO KNIGHTS': 'Castle & Medieval',
    'Space': 'Space & Sci-Fi', 'Aquazone': 'Space & Sci-Fi', 'Alpha Team': 'Space & Sci-Fi',
    'Exo-Force': 'Space & Sci-Fi', 'Power Miners': 'Space & Sci-Fi', 'Rock Raiders': 'Space & Sci-Fi',
    'Agents': 'Space & Sci-Fi', 'Ultra Agents': 'Space & Sci-Fi', 'Atlantis': 'Space & Sci-Fi',
    'Adventurers': 'Adventure', 'Indiana Jones': 'Adventure', "Pharaoh's Quest": 'Adventure',
    'Dino Attack': 'Adventure', 'Dino': 'Adventure', 'Western': 'Adventure',
    'SPEED CHAMPIONS': 'Racing & Vehicles', 'Racers': 'Racing & Vehicles',
    'World Racers': 'Racing & Vehicles', 'Speed Racer': 'Racing & Vehicles', 'Cars': 'Racing & Vehicles',
    'LEGENDS OF CHIMA': 'Fantasy', 'DREAMZzz': 'Fantasy', 'Hidden Side': 'Fantasy',
    'Monster Fighters': 'Fantasy',
    'The Hobbit and The Lord of the Rings': 'Middle-Earth',
    'Teenage Mutant Ninja Turtles': 'Licensed Entertainment',
    'SpongeBob SquarePants': 'Licensed Entertainment', 'Scooby-Doo': 'Licensed Entertainment',
    'The Simpsons': 'Licensed Entertainment', 'Ghostbusters': 'Licensed Entertainment',
    'Stranger Things': 'Licensed Entertainment', 'The Angry Birds Movie': 'Licensed Entertainment',
    'Trolls World Tour': 'Licensed Entertainment', 'The Incredibles': 'Licensed Entertainment',
    'Toy Story': 'Licensed Entertainment', 'Wednesday': 'Licensed Entertainment',
    'Wicked': 'Licensed Entertainment', 'Despicable Me and Minions': 'Licensed Entertainment',
    'The Lone Ranger': 'Licensed Entertainment', 'Prince of Persia': 'Licensed Entertainment',
    'Island Xtreme Stunts': 'Licensed Entertainment', 'The Powerpuff Girls': 'Licensed Entertainment',
    'Back to the Future': 'Licensed Entertainment', 'Friends TV Series': 'Licensed Entertainment',
    'Queer Eye': 'Licensed Entertainment', 'Lightyear': 'Licensed Entertainment',
    'Bluey': 'Licensed Entertainment', "Gabby's Dollhouse": 'Licensed Entertainment',
    'Dune': 'Licensed Entertainment', 'Star Trek': 'Licensed Entertainment',
    'Overwatch': 'Video Games', 'Fortnite': 'Video Games', 'Sonic the Hedgehog': 'Video Games',
    'Horizon': 'Video Games', 'The Legend of Zelda': 'Video Games', 'Animal Crossing': 'Video Games',
    'Collectible Minifigures': 'Collectible & Special', 'Holiday & Event': 'Collectible & Special',
    'Dimensions': 'Collectible & Special', 'Vidiyo': 'Collectible & Special', 'Games': 'Collectible & Special',
    'LEGO Ideas': 'LEGO Promotional', 'BrickLink Designer Program': 'LEGO Promotional',
    'LEGO Brand': 'LEGO Promotional', 'LEGOLAND': 'LEGO Promotional',
    'LEGOLAND Parks': 'LEGO Promotional', 'Promotional': 'LEGO Promotional',
    'Studios': 'LEGO Promotional', 'FIRST LEGO League': 'LEGO Promotional',
    'Educational & Dacta': 'Education & Technical', 'BIONICLE': 'Education & Technical',
    'Hero Factory': 'Education & Technical', 'Technic': 'Education & Technical',
    'Basic': 'Education & Technical', 'FreeStyle': 'Education & Technical',
}

def map_theme(theme):
    return MERGE_MAP.get(theme, theme)

# ============================================================
# TRANSFORMS
# ============================================================
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.RandAugment(num_ops=2, magnitude=9),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ============================================================
# DATASET
# ============================================================
class MinifigDataset(Dataset):
    def __init__(self, image_list, labels, transform=None):
        self.image_list = image_list
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = theme_path_map[img_name]
        img = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label, img_name

# ============================================================
# LOAD TRAINING DATA
# ============================================================
print("\nLoading training data...")
train_data = []
train_labels = []
class_names = set()

for img_name in valid_images:
    theme = map_theme(theme_map[img_name])
    class_names.add(theme)
    train_data.append(img_name)
    train_labels.append(theme)

class_names = sorted(list(class_names))
class_to_idx = {c: i for i, c in enumerate(class_names)}
train_labels_idx = [class_to_idx[c] for c in train_labels]

print(f"Classes: {len(class_names)}")
print(f"Training samples: {len(train_data)}")
print(f"Class distribution:\n{Counter([class_names[i] for i in train_labels_idx])}")

# ============================================================
# TENSORBOARD WRITER
# ============================================================
writer = SummaryWriter(LOG_DIR)

# ============================================================
# 1. VISUALIZE ORIGINAL IMAGES (before augmentation)
# ============================================================
print("\n1. Logging original images...")
val_dataset = MinifigDataset(train_data[:32], train_labels_idx[:32], transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

for batch_idx, (images, labels, names) in enumerate(val_loader):
    images = images.to(DEVICE)
    # Unnormalize for visualization
    images = images * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(DEVICE) + \
             torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(DEVICE)
    images = torch.clamp(images, 0, 1)

    grid = torch.clamp(images, 0, 1)
    labels_str = [f"{class_names[l.item()]}" for l in labels]
    writer.add_images(f'00_OriginalImages/Batch_{batch_idx}', grid, global_step=0)
    print(f"  Batch {batch_idx}: {labels_str[:4]}")

# ============================================================
# 2. VISUALIZE AUGMENTED IMAGES
# ============================================================
print("\n2. Logging augmented images...")
aug_dataset = MinifigDataset(train_data[:32], train_labels_idx[:32], transform=train_transform)
aug_loader = DataLoader(aug_dataset, batch_size=8, shuffle=False)

for batch_idx, (images, labels, names) in enumerate(aug_loader):
    images = images.to(DEVICE)
    # Unnormalize
    images = images * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(DEVICE) + \
             torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(DEVICE)
    images = torch.clamp(images, 0, 1)

    grid = torch.clamp(images, 0, 1)
    writer.add_images(f'01_AugmentedImages/Batch_{batch_idx}', grid, global_step=0)
    print(f"  Batch {batch_idx}: Augmented samples logged")

# ============================================================
# 3. COMPARE ORIGINAL VS AUGMENTED (side-by-side)
# ============================================================
print("\n3. Logging original vs augmented comparison...")
for idx in range(min(8, len(train_data))):
    orig_dataset = MinifigDataset([train_data[idx]], [train_labels_idx[idx]], transform=val_transform)
    aug_dataset = MinifigDataset([train_data[idx]], [train_labels_idx[idx]], transform=train_transform)

    orig_img = orig_dataset[0][0].unsqueeze(0)
    aug_imgs = torch.stack([aug_dataset[0][0] for _ in range(4)])

    # Unnormalize
    orig_img = orig_img * torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) + \
               torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    aug_imgs = aug_imgs * torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) + \
               torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)

    orig_img = torch.clamp(orig_img, 0, 1)
    aug_imgs = torch.clamp(aug_imgs, 0, 1)

    combined = torch.cat([orig_img, aug_imgs], dim=0)
    writer.add_images(f'02_Comparison/{class_names[train_labels_idx[idx]]}_Sample_{idx}',
                      combined, global_step=0)
    print(f"  Comparison {idx}: {class_names[train_labels_idx[idx]]}")

# ============================================================
# 4. CLASS DISTRIBUTION HISTOGRAM
# ============================================================
print("\n4. Logging class distribution...")
class_counts = Counter(train_labels)
classes_sorted = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)

fig, ax = plt.subplots(figsize=(14, 6))
class_names_sorted = [c[0] for c in classes_sorted]
counts_sorted = [c[1] for c in classes_sorted]
bars = ax.bar(range(len(class_names_sorted)), counts_sorted, color='steelblue', alpha=0.8)
ax.set_xticks(range(len(class_names_sorted)))
ax.set_xticklabels(class_names_sorted, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Number of Samples', fontsize=10)
ax.set_title('V3 Training Data: Class Distribution', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, (bar, count) in enumerate(zip(bars, counts_sorted)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
            str(count), ha='center', va='bottom', fontsize=8)

plt.tight_layout()
writer.add_figure('03_ClassDistribution/HistogramAllClasses', fig, global_step=0)
plt.close()
print(f"  Distribution: {len(class_names)} classes")

# ============================================================
# 5. AUGMENTATION VARIATIONS (single image, multiple augmentations)
# ============================================================
print("\n5. Logging augmentation variations...")
sample_idx = 0
sample_name = train_data[sample_idx]
sample_label_idx = train_labels_idx[sample_idx]
sample_label = class_names[sample_label_idx]

fig, axes = plt.subplots(2, 4, figsize=(12, 6))
fig.suptitle(f'V3 Augmentation Variations: {sample_label}', fontsize=12, fontweight='bold')

aug_dataset = MinifigDataset([sample_name] * 8, [sample_label_idx] * 8, transform=train_transform)

for i, ax in enumerate(axes.flat):
    aug_img = aug_dataset[i][0]
    # Unnormalize
    aug_img = aug_img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + \
              torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    aug_img = torch.clamp(aug_img, 0, 1).permute(1, 2, 0).numpy()
    ax.imshow(aug_img)
    ax.set_title(f'Augment {i+1}', fontsize=9)
    ax.axis('off')

plt.tight_layout()
writer.add_figure('04_AugmentationVariations/Single_Sample', fig, global_step=0)
plt.close()
print(f"  Variations shown for: {sample_label}")

# ============================================================
# 6. IMAGE STATISTICS
# ============================================================
print("\n6. Computing image statistics...")
all_means = []
all_stds = []
batch_size = 32

train_dataset = MinifigDataset(train_data, train_labels_idx, transform=val_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

for images, _, _ in train_loader:
    # Unnormalize first
    images = images * torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) + \
             torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)

    all_means.append(images.mean(dim=[0, 2, 3]))
    all_stds.append(images.std(dim=[0, 2, 3]))

mean_rgb = torch.stack(all_means).mean(dim=0)
std_rgb = torch.stack(all_stds).mean(dim=0)

writer.add_text('05_ImageStatistics/MeanRGB', f'R: {mean_rgb[0]:.4f}, G: {mean_rgb[1]:.4f}, B: {mean_rgb[2]:.4f}')
writer.add_text('05_ImageStatistics/StdRGB', f'R: {std_rgb[0]:.4f}, G: {std_rgb[1]:.4f}, B: {std_rgb[2]:.4f}')
print(f"  Mean RGB: {mean_rgb.numpy()}")
print(f"  Std RGB: {std_rgb.numpy()}")

# ============================================================
# 7. HYPERPARAMETERS INFO
# ============================================================
hparams_text = f"""
## V3 Model Configuration

**Data:**
- Total Samples: {len(train_data)}
- Classes: {len(class_names)}
- Image Size: {IMG_SIZE}x{IMG_SIZE}

**Augmentation Pipeline:**
- Resize: {IMG_SIZE + 32}x{IMG_SIZE + 32}
- RandomCrop: {IMG_SIZE}x{IMG_SIZE}
- RandomHorizontalFlip: p=0.5
- RandomRotation: 20°
- RandAugment: num_ops=2, magnitude=9
- ColorJitter: brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
- RandomPerspective: distortion_scale=0.2, p=0.3
- RandomErasing: p=0.2, scale=(0.02, 0.15)

**Normalization:**
- Mean: [0.485, 0.456, 0.406]
- Std: [0.229, 0.224, 0.225]

**Training:**
- Batch Size: 16
- Optimizer: AdamW
- LR: 3e-4
- Epochs: 20
- Loss: Focal, ArcFace, or SupCon
- Early Stopping: 5 epochs patience
"""
writer.add_text('06_HyperParameters/Configuration', hparams_text)

# ============================================================
# CLOSE & SUMMARY
# ============================================================
writer.close()

print("\n" + "="*80)
print("✅ TensorBoard visualization complete!")
print("="*80)
print(f"\n📊 Logs saved to: {LOG_DIR}")
print(f"\n🎯 To view results, run:")
print(f"   tensorboard --logdir {LOG_DIR}")
print(f"\n📈 Then open: http://localhost:6006")
print("\n" + "="*80)
