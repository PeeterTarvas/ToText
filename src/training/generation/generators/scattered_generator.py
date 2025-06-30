import random
from PIL import Image, ImageDraw

from ..base import BaseDataGenerator
from ..constants import IMG_SIZE, MIN_CHARS_PER_IMAGE, MAX_CHARS_PER_IMAGE


class ScatteredGenerator(BaseDataGenerator):
    """Generator for scattered character layouts"""

    def place_character_with_grid(self, canvas, char_img, occupied_grid, grid_size=10):
        """Place a character on the canvas avoiding overlaps"""
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
                # Mark as occupied
                for i in range(grid_x, min(grid_x + grid_w, IMG_SIZE // grid_size)):
                    for j in range(grid_y, min(grid_y + grid_h, IMG_SIZE // grid_size)):
                        occupied_grid[j][i] = True
                return x, y

        return None, None

    def generate(self, max_chars=None):
        """Generate an image with scattered character layout"""
        img = Image.new('L', (IMG_SIZE, IMG_SIZE), color=255)
        draw = ImageDraw.Draw(img)
        label_lines = []

        grid_size = 10
        grid_width = IMG_SIZE // grid_size + 1
        grid_height = IMG_SIZE // grid_size + 1
        occupied_grid = [[False] * grid_width for _ in range(grid_height)]

        chars_per_image = random.randint(MIN_CHARS_PER_IMAGE, max_chars)
        chars_placed = 0

        while chars_placed < chars_per_image:
            char_img, class_id = self.get_random_char_image()
            if char_img is None:
                continue

            w, h = char_img.size
            x, y = self.place_character_with_grid(img, char_img, occupied_grid)

            if x is None or y is None:
                continue

            img.paste(char_img, (x, y))

            x_center = (x + w / 2) / IMG_SIZE
            y_center = (y + h / 2) / IMG_SIZE
            width = w / IMG_SIZE
            height = h / IMG_SIZE
            label_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

            chars_placed += 1

        img = self.add_background_noise(img)
        return img, label_lines