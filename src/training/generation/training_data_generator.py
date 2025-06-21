import os
import random
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps, ImageFont, ImageDraw, ImageFilter, ImageEnhance
import torch
from torchvision import datasets, transforms
import textwrap

LOREM_IPSUM = """
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. 
Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. 
Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. 
Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. 
Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. 
Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. 
Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. 
Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. 
Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. 
Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. 
Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. 
Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. 
Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. 
Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. 
Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. 
Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. 
Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. 
Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. 
Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. 
Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. 
Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. 
Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
"""

# Constants
IMG_SIZE = 640
MIN_CHARS_PER_IMAGE = 30
MAX_CHARS_PER_IMAGE = 200
TOTAL_IMAGES = 10
TRAIN_SPLIT = 0.8
DATA_ROOT = Path("synthetic_emnist_yolo")
DOC_MARGIN = 40
MIN_DOC_FONT_SIZE = 10
MAX_DOC_FONT_SIZE = 20
DOC_LINE_SPACING_FACTOR = 1.3
ROTATION_RANGE = (-15, 15)
SCALE_RANGE = (0.7, 1.3)
CHAR_SPACING_RANGE = (5, 20)
LINE_SPACING_RANGE = (30, 50)
FONT_SIZE_RANGE = (24, 36)

FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
]

CUSTOM_FONT_DIR = Path("src/training/resources/fonts")
if CUSTOM_FONT_DIR.exists():
    custom_fonts = list(CUSTOM_FONT_DIR.glob("**/*.ttf")) + list(CUSTOM_FONT_DIR.glob("**/*.otf"))
    FONT_PATHS += [str(font_path) for font_path in custom_fonts]


random.seed(42)
os.makedirs(DATA_ROOT, exist_ok=True)
for split in ['train', 'val']:
    (DATA_ROOT / f"images/{split}").mkdir(parents=True, exist_ok=True)
    (DATA_ROOT / f"labels/{split}").mkdir(parents=True, exist_ok=True)

transform = transforms.Compose([transforms.ToTensor()])
emnist = datasets.EMNIST(root=DATA_ROOT, split='balanced', download=True, train=True, transform=transform)
class_map = list(range(94))
samples_by_class = {i: [] for i in class_map}
for img, label in emnist:
    samples_by_class[label].append(img)

CHAR_MAP = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!?.,:;'\"()[]{}<>@#$%^&*+-=/\\|_"

def get_random_font(size=None):
    font_path = random.choice(FONT_PATHS)
    if size is None:
        size = random.randint(*FONT_SIZE_RANGE)
    try:
        return ImageFont.truetype(font_path, size)
    except OSError:
        return ImageFont.load_default()

def augment_image(img):
    angle = random.uniform(*ROTATION_RANGE)
    img = img.rotate(angle, expand=1, fillcolor=255)
    scale = random.uniform(*SCALE_RANGE)
    w, h = img.size
    img = img.resize((max(10, int(w * scale)), max(10, int(h * scale))))
    return img

def generate_font_char(char, font=None):
    if font is None:
        font = get_random_font()
    bbox = font.getbbox(char)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    padding = 5
    img = Image.new("L", (width + 2 * padding, height + 2 * padding), 255)
    draw = ImageDraw.Draw(img)
    draw.text((padding - bbox[0], padding - bbox[1]), char, font=font, fill=0)
    img = augment_image(img)
    return img

def place_character_with_grid(canvas, char_img, occupied_grid, grid_size=10):
    w, h = char_img.size
    grid_w = (w // grid_size) + 1
    grid_h = (h // grid_size) + 1
    max_attempts = 100
    for _ in range(max_attempts):
        max_x = IMG_SIZE - w
        max_y = IMG_SIZE - h
        if max_x < 0 or max_y < 0:
            return None, None
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)
        grid_x = x // grid_size
        grid_y = y // grid_size
        overlap = False
        for i in range(max(0, grid_x - grid_w), min(grid_x + grid_w, IMG_SIZE // grid_size)):
            for j in range(max(0, grid_y - grid_h), min(grid_y + grid_h, IMG_SIZE // grid_size)):
                if occupied_grid[j][i]:
                    overlap = True
                    break
            if overlap:
                break
        if not overlap:
            for i in range(grid_x, min(grid_x + grid_w, IMG_SIZE // grid_size)):
                for j in range(grid_y, min(grid_y + grid_h, IMG_SIZE // grid_size)):
                    occupied_grid[j][i] = True
            return x, y
    return None, None

def generate_text_line(canvas, draw, label_lines, y_start, occupied_grid, font=None):
    x = random.randint(20, 50)
    y = y_start
    char_spacing = random.randint(*CHAR_SPACING_RANGE)
    line_height = random.randint(*LINE_SPACING_RANGE)
    while x < IMG_SIZE - 50:
        if random.random() < 0.5:
            char = random.choice(CHAR_MAP)
            class_id = CHAR_MAP.index(char) % len(class_map)
            char_img = generate_font_char(char, font)
        else:
            class_id = random.choice(class_map)
            if not samples_by_class[class_id]:
                continue
            char_img = transforms.ToPILImage()(random.choice(samples_by_class[class_id]).squeeze(0))
            char_img = ImageOps.invert(char_img)
            char_img = char_img.resize((32, 32))
            char_img = augment_image(char_img)
        w, h = char_img.size
        if x + w > IMG_SIZE - 10:
            break
        grid_x = x // 10
        grid_y = y // 10
        grid_w = (w // 10) + 1
        grid_h = (h // 10) + 1
        conflict = False
        for i in range(grid_x, min(grid_x + grid_w, IMG_SIZE // 10)):
            for j in range(grid_y, min(grid_y + grid_h, IMG_SIZE // 10)):
                if occupied_grid[j][i]:
                    conflict = True
                    break
            if conflict:
                break
        if conflict:
            x += char_spacing
            continue
        for i in range(grid_x, min(grid_x + grid_w, IMG_SIZE // 10)):
            for j in range(grid_y, min(grid_y + grid_h, IMG_SIZE // 10)):
                occupied_grid[j][i] = True
        canvas.paste(char_img, (x, y))
        x_center = (x + w / 2) / IMG_SIZE
        y_center = (y + h / 2) / IMG_SIZE
        width = w / IMG_SIZE
        height = h / IMG_SIZE
        label_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        x += w + char_spacing
    return y + line_height

def generate_random_text_for_document():
    paragraphs = LOREM_IPSUM.strip().split('\n\n')
    random.shuffle(paragraphs)
    full_text = ' '.join(paragraphs)
    words = full_text.split()

def generate_document_page():
    """Generate an A4-style document page with structured text lines"""
    img = Image.new('L', (IMG_SIZE, IMG_SIZE), color=255)
    draw = ImageDraw.Draw(img)
    label_lines = []

    font_size = random.randint(MIN_DOC_FONT_SIZE, MAX_DOC_FONT_SIZE)
    font = get_random_font(size=font_size)
    line_spacing = int(font_size * DOC_LINE_SPACING_FACTOR)

    # Prepare text content
    paragraphs = LOREM_IPSUM.strip().split('\n\n')
    random.shuffle(paragraphs)
    full_text = ' '.join(paragraphs)
    words = full_text.split()

    # Generate text lines
    y = DOC_MARGIN
    max_width = IMG_SIZE - 2 * DOC_MARGIN

    while y < IMG_SIZE - DOC_MARGIN and words:
        # Build line
        line_words = []
        line_width = 0

        while words and line_width < max_width:
            word = words.pop(0)
            word_width = font.getlength(word + ' ')

            if not line_words or line_width + word_width <= max_width:
                line_words.append(word)
                line_width += word_width
            else:
                words.insert(0, word)
                break

        if not line_words:
            break

        line_text = ' '.join(line_words)

        x = DOC_MARGIN
        for char in line_text:
            if char not in CHAR_MAP:
                # Skip unsupported characters
                continue

            bbox = font.getbbox(char)
            char_width = bbox[2] - bbox[0]
            char_height = bbox[3] - bbox[1]

            # Draw character
            draw.text((x, y), char, font=font, fill=0)

            # Calculate YOLO coordinates
            x_center = (x + char_width / 2) / IMG_SIZE
            y_center = (y + char_height / 2) / IMG_SIZE
            width = char_width / IMG_SIZE
            height = char_height / IMG_SIZE

            # Get class ID
            class_id = CHAR_MAP.index(char) % len(class_map)
            label_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

            # Move to next character position
            x += char_width

        # Move to next line
        y += line_spacing

    return img, label_lines

def add_background_noise(image):
    arr = np.array(image).astype(np.float32)
    noise = np.random.normal(0, random.uniform(2, 8), arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    noisy_img = Image.fromarray(arr)
    if random.random() > 0.7:
        noisy_img = ImageEnhance.Brightness(noisy_img).enhance(random.uniform(0.8, 1.2))
    if random.random() > 0.7:
        noisy_img = ImageEnhance.Contrast(noisy_img).enhance(random.uniform(0.8, 1.2))
    if random.random() > 0.5:
        noisy_img = noisy_img.filter(ImageFilter.GaussianBlur(random.uniform(0.1, 1.0)))
    return noisy_img

indices = list(range(TOTAL_IMAGES))
random.shuffle(indices)
train_cutoff = int(TRAIN_SPLIT * TOTAL_IMAGES)

for idx in range(TOTAL_IMAGES):
    split = 'train' if idx < train_cutoff else 'val'
    generation_type = random.choices(['line', 'scattered', 'document'], weights=[0.33, 0.33, 0.34])[0]

    if generation_type == 'document':
        # Generate document-style page
        img, label_lines = generate_document_page()
    else:
        # Create blank image and grid
        img = Image.new('L', (IMG_SIZE, IMG_SIZE), color=255)
        draw = ImageDraw.Draw(img)
        label_lines = []
        grid_size = 10
        grid_width = IMG_SIZE // grid_size + 1
        grid_height = IMG_SIZE // grid_size + 1
        occupied_grid = [[False] * grid_width for _ in range(grid_height)]
        chars_per_image = random.randint(MIN_CHARS_PER_IMAGE, MAX_CHARS_PER_IMAGE)

        if generation_type == 'line':
            # Generate line characters
            y_pos = 20
            font = get_random_font() if random.random() < 0.8 else None
            while y_pos < IMG_SIZE - 50 and len(label_lines) < chars_per_image:
                y_pos = generate_text_line(img, draw, label_lines, y_pos, occupied_grid, font)
        else:
            # Generate scattered characters
            chars_placed = 0
            while chars_placed < chars_per_image:
                if random.random() < 0.5:
                    char = random.choice(CHAR_MAP)
                    class_id = CHAR_MAP.index(char) % len(class_map)
                    char_img = generate_font_char(char)
                else:
                    class_id = random.choice(class_map)
                    if not samples_by_class[class_id]:
                        continue
                    char_img = transforms.ToPILImage()(random.choice(samples_by_class[class_id]).squeeze(0))
                    char_img = ImageOps.invert(char_img)
                    char_img = char_img.resize((32, 32))
                    char_img = augment_image(char_img)
                w, h = char_img.size
                x, y = place_character_with_grid(img, char_img, occupied_grid)
                if x is None or y is None:
                    continue
                img.paste(char_img, (x, y))
                x_center = (x + w / 2) / IMG_SIZE
                y_center = (y + h / 2) / IMG_SIZE
                width = w / IMG_SIZE
                height = h / IMG_SIZE
                label_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                chars_placed += 1

    # Add noise and save
    img = add_background_noise(img)
    img_name = f"img_{idx:04d}.png"
    label_name = f"img_{idx:04d}.txt"
    img.save(DATA_ROOT / f"images/{split}" / img_name)
    with open(DATA_ROOT / f"labels/{split}" / label_name, 'w') as f:
        f.write("\n".join(label_lines))
    print(f"Generated {generation_type} image {idx+1}/{TOTAL_IMAGES} with {len(label_lines)} characters")

print("Dataset generation complete.")