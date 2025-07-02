import wikipedia
import random
from PIL import Image, ImageDraw
from .document_generator import DocumentGenerator
from ..constants import (IMG_SIZE, MIN_CHARS_PER_IMAGE, MAX_CHARS_PER_IMAGE,
                         MIN_DOC_FONT_SIZE, MAX_DOC_FONT_SIZE, DOC_LINE_SPACING_FACTOR
                         ,DOC_MARGIN, CHAR_MAP)


class WikipediaDocumentGenerator(DocumentGenerator):

    def __init__(self, samples_by_class, class_map, page_title=None):
        super().__init__(samples_by_class, class_map)
        self.page_title = page_title or "Artificial intelligence"

    def get_wikipedia_text(self):
        self.page_title = wikipedia.random(pages=1)
        try:
            text = wikipedia.page(self.page_title).content
        except Exception:
            text = wikipedia.summary("Artificial intelligence")
        wiki_text = self.filter_unsupported_chars(text)
        return wiki_text.strip().replace('\n', ' ')[:400]

    def generate(self, max_chars=None):
        """Generate an image using Wikipedia text"""
        text = self.get_wikipedia_text()

        paragraphs = []
        words = text.split()
        idx = 0
        while idx < len(words):
            para_len = random.randint(20, 60)
            para_words = words[idx:idx + para_len]
            paragraph = ' '.join(para_words)
            paragraphs.append(paragraph)
            idx += para_len

        wiki_text = '\n\n'.join(paragraphs)
        self.full_text_override = wiki_text

        return self._generate_from_text(wiki_text)

    def _generate_from_text(self, full_text):
        """Helper to reuse the original layout rendering logic"""
        img = Image.new('L', (IMG_SIZE, IMG_SIZE), color=255)
        draw = ImageDraw.Draw(img)
        label_lines = []

        font_size = random.randint(MIN_DOC_FONT_SIZE, MAX_DOC_FONT_SIZE)
        font = self.get_random_font(size=font_size)
        line_spacing = int(font_size * DOC_LINE_SPACING_FACTOR)
        max_width = IMG_SIZE - 2 * DOC_MARGIN

        words = full_text.replace('\n\n', ' \n\n ').split()
        y = DOC_MARGIN
        line_count = 0
        paragraph_break_interval = random.randint(3, 6)

        while y < IMG_SIZE - DOC_MARGIN and words:
            line_words = []
            line_width = 0
            while words and line_width < max_width:
                word = words.pop(0)
                if word == '\n\n':
                    if line_words:
                        break
                    words.pop(0) if words and words[0] == '\n\n' else None
                    continue

                prefix = " " if line_words else ""
                word_width = font.getlength(prefix + word)

                if line_words and line_width + word_width > max_width:
                    words.insert(0, word)
                    break

                line_words.append(word)
                line_width += word_width

            if not line_words:
                continue

            line_text = " ".join(line_words)
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

                x_center = (x + char_width / 2) / IMG_SIZE
                y_center = (y + char_height / 2) / IMG_SIZE
                width = char_width / IMG_SIZE
                height = char_height / IMG_SIZE

                x_center = max(0.0, min(1.0, x_center))
                y_center = max(0.0, min(1.0, y_center))
                width = max(0.0, min(1.0 - x_center, width))
                height = max(0.0, min(1.0 - y_center, height))

                class_id = CHAR_MAP.index(char) % len(self.class_map)

                label_lines.append(
                    f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                )

                x += char_width

            y += line_spacing
            line_count += 1
            if line_count >= paragraph_break_interval:
                line_count = 0
                y += line_spacing
                if y >= IMG_SIZE - DOC_MARGIN:
                    break

        img = self.add_background_noise(img)
        return img, label_lines

    def filter_unsupported_chars(self, text):
        return ''.join(c for c in text if c in CHAR_MAP or c.isspace())

    def set_title(self, title):
        self.title = title