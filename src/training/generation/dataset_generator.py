import os
import random
from torchvision import datasets, transforms
import uuid

from .constants import DATA_ROOT, TOTAL_IMAGES, TRAIN_SPLIT
from .generators import LineGenerator, ScatteredGenerator, DocumentGenerator


class SyntheticDatasetGenerator:
    """Main class that orchestrates the data generation process"""

    def __init__(self):
        random.seed(42)

        self._setup_directories()

        self._load_emnist()

        self.line_generator = LineGenerator(self.samples_by_class, self.class_map)
        self.scattered_generator = ScatteredGenerator(self.samples_by_class, self.class_map)
        self.document_generator = DocumentGenerator(self.samples_by_class, self.class_map)

        self.generators = {
            'line': self.line_generator,
            'scattered': self.scattered_generator,
            'document': self.document_generator
        }

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

        self.class_map = list(range(94))
        self.samples_by_class = {i: [] for i in self.class_map}

        for img, label in emnist:
            self.samples_by_class[label].append(img)

    def generate_dataset(self):
        """Generate the complete dataset"""
        indices = list(range(TOTAL_IMAGES))
        random.shuffle(indices)
        train_cutoff = int(TRAIN_SPLIT * TOTAL_IMAGES)

        for idx in range(TOTAL_IMAGES):
            split = 'train' if idx < train_cutoff else 'val'

            generation_type = random.choices(
                ['line', 'scattered', 'document'],
                weights=[0.33, 0.33, 0.34]
            )[0]

            generator = self.generators[generation_type]
            img, label_lines = generator.generate()

            uuid_name_tag = uuid.uuid4()
            img_name = f"{generation_type}_{idx}_{uuid_name_tag}.png"
            label_name = f"{generation_type}_{idx}_{uuid_name_tag}.txt"

            img.save(DATA_ROOT / f"images/{split}" / img_name)

            with open(DATA_ROOT / f"labels/{split}" / label_name, 'w') as f:
                f.write("\n".join(label_lines))

            print(f"Generated {generation_type} image {idx + 1}/{TOTAL_IMAGES} with {len(label_lines)} characters")

        print("Dataset generation complete.")