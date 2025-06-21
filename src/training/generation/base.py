import random
import numpy as np
from abc import ABC, abstractmethod
from PIL import Image, ImageOps, ImageFont, ImageDraw, ImageFilter, ImageEnhance
from torchvision import transforms

from .constants import (
    FONT_PATHS, FONT_SIZE_RANGE, ROTATION_RANGE, SCALE_RANGE, CHAR_MAP
)


class BaseDataGenerator(ABC):
    """Base class for all data generation methods with common functionality"""

    def __init__(self, samples_by_class, class_map):
        self.samples_by_class = samples_by_class
        self.class_map = class_map
        self.transform = transforms.ToPILImage()

    @staticmethod
    def get_random_font(size=None):
        """Get a random font with specified or random size"""
        font_path = random.choice(FONT_PATHS)
        if size is None:
            size = random.randint(*FONT_SIZE_RANGE)
        try:
            return ImageFont.truetype(font_path, size)
        except OSError:
            return ImageFont.load_default()

    @staticmethod
    def augment_image(img):
        """Apply augmentation to an image"""
        angle = random.uniform(*ROTATION_RANGE)
        img = img.rotate(angle, expand=1, fillcolor=255)
        scale = random.uniform(*SCALE_RANGE)
        w, h = img.size
        img = img.resize((max(10, int(w * scale)), max(10, int(h * scale))))
        return img

    def generate_font_char(self, char, font=None):
        """Generate a character image using a font"""
        if font is None:
            font = self.get_random_font()
        bbox = font.getbbox(char)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        padding = 5
        img = Image.new("L", (width + 2 * padding, height + 2 * padding), 255)
        draw = ImageDraw.Draw(img)
        draw.text((padding - bbox[0], padding - bbox[1]), char, font=font, fill=0)
        img = self.augment_image(img)
        return img

    def get_random_char_image(self, font=None):
        """Get a random character image from font or EMNIST"""
        if random.random() < 0.5:
            char = random.choice(CHAR_MAP)
            class_id = CHAR_MAP.index(char) % len(self.class_map)
            char_img = self.generate_font_char(char, font)
        else:
            class_id = random.choice(self.class_map)
            if not self.samples_by_class[class_id]:
                return None, None
            char_img = self.transform(random.choice(self.samples_by_class[class_id]).squeeze(0))
            char_img = ImageOps.invert(char_img)
            char_img = char_img.resize((32, 32))
            char_img = self.augment_image(char_img)
        return char_img, class_id

    @staticmethod
    def add_background_noise(image):
        """Add background noise to an image"""
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

    @abstractmethod
    def generate(self):
        """Generate an image with labels. Must be implemented by subclasses."""
        pass