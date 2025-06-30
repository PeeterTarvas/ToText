import os
import random
from torchvision import datasets, transforms
import uuid
import numpy as np
from PIL import Image
from .augmentors import Augmenter
import multiprocessing
from functools import partial

from .constants import DATA_ROOT, TOTAL_IMAGES, TRAIN_SPLIT, CHAR_MAP, MAX_CHARS_PER_IMAGE
from .generators import LineGenerator, ScatteredGenerator, DocumentGenerator, WikipediaDocumentGenerator

def _generate_one_image(idx, total_images, train_split, samples_by_class, class_map):
    """Core image generation function usable in both single/multi-process modes"""
    split = 'train' if idx < int(train_split * total_images) else 'val'
    difficulty = idx / total_images

    # Progressive difficulty configuration
    if difficulty < 0.1:
        generation_type = 'line'
        num_augmentations = 1
        max_chars = MAX_CHARS_PER_IMAGE * 0.1
    elif difficulty < 0.3:
        generation_type = random.choice(['line', 'scattered'])
        num_augmentations = random.randint(1, 2)
        max_chars = MAX_CHARS_PER_IMAGE * 0.3
    elif difficulty < 0.55:
        generation_type = random.choice(['line', 'scattered', 'document'])
        num_augmentations = random.randint(1, 3)
        max_chars = MAX_CHARS_PER_IMAGE * 0.6
    else:
        generation_type = random.choice(['line', 'scattered', 'document', 'wikipedia'])
        num_augmentations = random.randint(2, 4)
        max_chars = MAX_CHARS_PER_IMAGE

    # Create generators locally (ensures thread safety)
    generators = {
        'line': LineGenerator(samples_by_class, class_map),
        'scattered': ScatteredGenerator(samples_by_class, class_map),
        'document': DocumentGenerator(samples_by_class, class_map),
        'wikipedia': WikipediaDocumentGenerator(samples_by_class, class_map)
    }

    augmenter = Augmenter()
    generator = generators[generation_type]

    # Generate and augment image
    img, label_lines = generator.generate(max_chars=max_chars)
    img = img.convert('RGB')

    if random.random() < 0.5:
        img_np = np.array(img)
        img_np = augmenter.random_augment(img_np, num_augmentations=num_augmentations)
        img = Image.fromarray(img_np)

    # Save results
    uuid_name_tag = uuid.uuid4()
    img_name = f"{generation_type}_{idx}_{uuid_name_tag}.png"
    label_name = f"{generation_type}_{idx}_{uuid_name_tag}.txt"

    img.save(DATA_ROOT / f"images/{split}" / img_name)

    with open(DATA_ROOT / f"labels/{split}" / label_name, 'w') as f:
        f.write("\n".join(label_lines))

    print(f"Generated {generation_type} image {idx + 1}/{total_images} with {len(label_lines)} characters")
    return True



class SyntheticDatasetGenerator:
    """Main class that orchestrates the data generation process"""

    def __init__(self):
        random.seed(42)

        self._setup_directories()

        self._load_emnist()

        self.line_generator = LineGenerator(self.samples_by_class, self.class_map)
        self.scattered_generator = ScatteredGenerator(self.samples_by_class, self.class_map)
        self.document_generator = DocumentGenerator(self.samples_by_class, self.class_map)
        self.wikipedia_generator = WikipediaDocumentGenerator(self.samples_by_class, self.class_map)

        self.generators = {
            'line': self.line_generator,
            'scattered': self.scattered_generator,
            'document': self.document_generator,
            'wikipedia': self.wikipedia_generator
        }

        self.augmenter = Augmenter()

    def _setup_directories(self):
        """Create necessary directories"""
        os.makedirs(DATA_ROOT, exist_ok=True)
        for split in ['train', 'val']:
            (DATA_ROOT / f"images/{split}").mkdir(parents=True, exist_ok=True)
            (DATA_ROOT / f"labels/{split}").mkdir(parents=True, exist_ok=True)

    def _load_emnist(self):
        """Load EMNIST dataset and organize by class"""
        transform = transforms.Compose([transforms.ToTensor()])
        emnist = datasets.EMNIST(root=DATA_ROOT, split='balanced', download=True, train=True, transform=transform)

        self.class_map = list(range(len(CHAR_MAP)))
        self.samples_by_class = {i: [] for i in self.class_map}

        for img, label in emnist:
            self.samples_by_class[label].append(img)

    def generate_dataset(self):
        """Generate dataset in single-process mode with progressive difficulty"""
        indices = list(range(TOTAL_IMAGES))
        random.shuffle(indices)

        for idx in indices:
            _generate_one_image(
                idx,
                TOTAL_IMAGES,
                TRAIN_SPLIT,
                self.samples_by_class,
                self.class_map
            )
        print("Dataset generation complete.")

    def generate_dataset_multi(self, threads=4):
        """Generate dataset using multiprocessing with progressive difficulty"""
        indices = list(range(TOTAL_IMAGES))
        random.shuffle(indices)

        worker_func = partial(
            _generate_one_image,
            total_images=TOTAL_IMAGES,
            train_split=TRAIN_SPLIT,
            samples_by_class=self.samples_by_class,
            class_map=self.class_map
        )

        with multiprocessing.Pool(processes=threads) as pool:
            pool.map(worker_func, indices)

        print("Multiprocess dataset generation complete.")
