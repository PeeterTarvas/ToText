import os
import random
from pathlib import Path
from PIL import Image, ImageOps, ImageFont, ImageDraw, ImageFilter
import torch
from torchvision import datasets, transforms

# Configuration
IMG_SIZE = 640
MIN_CHARS_PER_IMAGE = 50
MAX_MIN_CHARS_PER_IMAGE = 200
TOTAL_IMAGES = 10
TRAIN_SPLIT = 0.8
DATA_ROOT = Path("synthetic_emnist_yolo")

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"  # Adjust to your system
USE_LINE_LAYOUT_PROB = 0.5
USE_FONT_CHAR_PROB = 0.3
ROTATION_RANGE = (-10, 10)
SCALE_RANGE = (0.8, 1.2)

random.seed(42)
os.makedirs(DATA_ROOT, exist_ok=True)
for split in ['train', 'val']:
    (DATA_ROOT / f"images/{split}").mkdir(parents=True, exist_ok=True)
    (DATA_ROOT / f"labels/{split}").mkdir(parents=True, exist_ok=True)

transform = transforms.Compose([transforms.ToTensor()])
emnist = datasets.EMNIST(root=DATA_ROOT, split='balanced', download=True, train=True, transform=transform)
class_map = list(range(47))
samples_by_class = {i: [] for i in class_map}
for img, label in emnist:
    samples_by_class[label].append(img)

# Augment and paste character
def augment_image(img):
    angle = random.uniform(*ROTATION_RANGE)
    scale = random.uniform(*SCALE_RANGE)
    img = img.rotate(angle, expand=1, fillcolor=255)
    w, h = img.size
    img = img.resize((int(w * scale), int(h * scale)))
    return img

# Create character using font
def generate_font_char(label_char):
    img = Image.new("L", (32, 32), 255)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT_PATH, 28)
    draw.text((2, 0), label_char, font=font, fill=0)
    return img

# Paste character and return YOLO label
def paste_character(canvas, char_img, label):
    char_img = augment_image(char_img)
    w, h = char_img.size
    max_x = IMG_SIZE - w
    max_y = IMG_SIZE - h
    x = random.randint(0, max_x)
    y = random.randint(0, max_y)
    canvas.paste(char_img, (x, y))
    x_center = (x + w / 2) / IMG_SIZE
    y_center = (y + h / 2) / IMG_SIZE
    return f"{label} {x_center:.6f} {y_center:.6f} {w / IMG_SIZE:.6f} {h / IMG_SIZE:.6f}"

# Simulate a line of characters
def generate_text_line(canvas, label_lines, y_start):
    x = 20
    y = y_start
    while x < IMG_SIZE - 40:
        class_id = random.choice(class_map)
        label_char = chr(65 + class_id % 26)
        if random.random() < USE_FONT_CHAR_PROB:
            char_img = generate_font_char(label_char)
        else:
            if samples_by_class[class_id]:
                char_img = transforms.ToPILImage()(random.choice(samples_by_class[class_id]).squeeze(0))
                char_img = ImageOps.invert(char_img)
                char_img = char_img.resize((32, 32))
            else:
                continue

        char_img = augment_image(char_img)
        w, h = char_img.size
        if x + w > IMG_SIZE - 20:
            break
        canvas.paste(char_img, (x, y))
        x_center = (x + w / 2) / IMG_SIZE
        y_center = (y + h / 2) / IMG_SIZE
        label_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w / IMG_SIZE:.6f} {h / IMG_SIZE:.6f}")
        x += w + random.randint(5, 15)
    return y + 40

# Add synthetic background noise
def add_background_noise(image):
    noise = Image.effect_noise((IMG_SIZE, IMG_SIZE), random.uniform(2, 5)).convert("L")
    noisy = Image.blend(image, noise, alpha=0.1)
    return noisy.filter(ImageFilter.GaussianBlur(radius=0.5))

# Dataset generation
indices = list(range(TOTAL_IMAGES))
random.shuffle(indices)
train_cutoff = int(TRAIN_SPLIT * TOTAL_IMAGES)

for idx in range(TOTAL_IMAGES):
    split = 'train' if idx < train_cutoff else 'val'
    img = Image.new('L', (IMG_SIZE, IMG_SIZE), color=255)
    label_lines = []

    if random.random() < USE_LINE_LAYOUT_PROB:
        y_cursor = 20
        while y_cursor < IMG_SIZE - 40:
            y_cursor = generate_text_line(img, label_lines, y_cursor)
    else:
        chars_per_image = random.randint(MIN_CHARS_PER_IMAGE, MAX_MIN_CHARS_PER_IMAGE)
        for _ in range(chars_per_image):
            class_id = random.choice(class_map)
            label_char = chr(65 + class_id % 26)

            if random.random() < USE_FONT_CHAR_PROB:
                char_img = generate_font_char(label_char)
            else:
                if samples_by_class[class_id]:
                    char_img = transforms.ToPILImage()(random.choice(samples_by_class[class_id]).squeeze(0))
                    char_img = ImageOps.invert(char_img)
                    char_img = char_img.resize((32, 32))
                else:
                    continue
            label_line = paste_character(img, char_img, class_id)
            label_lines.append(label_line)

    img = add_background_noise(img)

    img_name = f"img_{idx:04d}.png"
    label_name = f"img_{idx:04d}.txt"
    img.save(DATA_ROOT / f"images/{split}" / img_name)
    with open(DATA_ROOT / f"labels/{split}" / label_name, 'w') as f:
        f.write("\n".join(label_lines))

print("Dataset generation complete.")
