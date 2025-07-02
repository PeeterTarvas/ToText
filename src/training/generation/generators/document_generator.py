import random
from PIL import Image, ImageDraw

from ..base import BaseDataGenerator
from ..constants import (
    IMG_SIZE, DOC_MARGIN, MIN_DOC_FONT_SIZE, MAX_DOC_FONT_SIZE,
    DOC_LINE_SPACING_FACTOR, CHAR_MAP
)


class DocumentGenerator(BaseDataGenerator):
    """Generator for document-style layouts with random text"""

    def generate(self, max_chars=None):
        """Generate an A4-style document page with random text lines"""
        img = Image.new('L', (IMG_SIZE, IMG_SIZE), color=255)
        draw = ImageDraw.Draw(img)
        label_lines = []

        font_size = random.randint(MIN_DOC_FONT_SIZE, MAX_DOC_FONT_SIZE)
        font = self.get_random_font(size=font_size)
        line_spacing = int(font_size * DOC_LINE_SPACING_FACTOR)
        max_width = IMG_SIZE - 5 * DOC_MARGIN

        paragraphs = []
        num_paragraphs = random.randint(3, 8)  # Number of paragraphs
        for _ in range(num_paragraphs):
            num_lines = random.randint(3, 8)  # Lines per paragraph
            paragraph_lines = []
            for _ in range(num_lines):
                # Generate line with 5-12 random "words"
                words = [
                    ''.join(random.choices(CHAR_MAP, k=random.randint(1, 8)))
                    for _ in range(random.randint(5, 12))
                ]
                paragraph_lines.append(' '.join(words))
            paragraphs.append('\n'.join(paragraph_lines))
        full_text = '\n\n'.join(paragraphs)
        words = full_text.replace('\n\n', ' \n\n ').split()  # Preserve paragraph breaks

        y = DOC_MARGIN
        line_count = 0
        paragraph_break_interval = random.randint(3, 6)  # Lines before paragraph break

        while y < IMG_SIZE - DOC_MARGIN and words:
            line_words = []
            line_width = 0

            while words and line_width < max_width:
                word = words.pop(0)

                # Handle paragraph breaks
                if word == '\n\n':
                    if line_words:  # Finish current line before break
                        break
                    # Skip to next line if empty
                    words.pop(0) if words and words[0] == '\n\n' else None
                    continue

                # Calculate word width (add space if not first word)
                prefix = " " if line_words else ""
                word_width = font.getlength(prefix + word)

                if line_words and line_width + word_width > max_width:
                    words.insert(0, word)  # Requeue word for next line
                    break

                line_words.append(word)
                line_width += word_width

            if not line_words:
                continue  # Skip empty lines

            line_text = " ".join(line_words)

            # Draw characters
            x = DOC_MARGIN
            for char in line_text:
                if char == ' ':
                    x += font.getlength(' ')
                    continue
                if char not in CHAR_MAP:
                    continue

                bbox = font.getbbox(char)
                char_width = bbox[2] - bbox[0]
                char_height = bbox[3] - bbox[1]

                draw.text((x, y), char, font=font, fill=0)

                # Calculate bounding box (YOLO format)
                x_center = (x + char_width / 2) / IMG_SIZE
                y_center = (y + char_height / 2) / IMG_SIZE
                width = char_width / IMG_SIZE
                height = char_height / IMG_SIZE

                class_id = CHAR_MAP.index(char) % len(self.class_map)
                label_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

                x += char_width

            # Update line position and count
            y += line_spacing
            line_count += 1

            # Add paragraph break
            if line_count >= paragraph_break_interval:
                line_count = 0
                y += line_spacing  # Extra space for paragraph
                if y >= IMG_SIZE - DOC_MARGIN:
                    break

        img = self.add_background_noise(img)
        return img, label_lines