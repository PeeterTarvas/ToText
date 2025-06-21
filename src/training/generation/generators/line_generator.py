import random
from PIL import Image, ImageDraw

from ..base import BaseDataGenerator
from ..constants import (
    IMG_SIZE, MIN_CHARS_PER_IMAGE, MAX_CHARS_PER_IMAGE,
    CHAR_SPACING_RANGE, LINE_SPACING_RANGE
)


class LineGenerator(BaseDataGenerator):
    """Generator for line-based character layouts"""

    def generate_text_line(self, canvas, draw, label_lines, y_start, occupied_grid, font=None):
        """Generate a single line of text"""
        x = random.randint(20, 50)
        y = y_start
        char_spacing = random.randint(*CHAR_SPACING_RANGE)
        line_height = random.randint(*LINE_SPACING_RANGE)

        while x < IMG_SIZE - 50:
            char_img, class_id = self.get_random_char_image(font)
            if char_img is None:
                continue

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

    def generate(self):
        """Generate an image with line-based character layout"""
        img = Image.new('L', (IMG_SIZE, IMG_SIZE), color=255)
        draw = ImageDraw.Draw(img)
        label_lines = []

        grid_size = 10
        grid_width = IMG_SIZE // grid_size + 1
        grid_height = IMG_SIZE // grid_size + 1
        occupied_grid = [[False] * grid_width for _ in range(grid_height)]

        chars_per_image = random.randint(MIN_CHARS_PER_IMAGE, MAX_CHARS_PER_IMAGE)

        y_pos = 20
        font = self.get_random_font() if random.random() < 0.8 else None

        while y_pos < IMG_SIZE - 50 and len(label_lines) < chars_per_image:
            y_pos = self.generate_text_line(img, draw, label_lines, y_pos, occupied_grid, font)

        img = self.add_background_noise(img)
        return img, label_lines