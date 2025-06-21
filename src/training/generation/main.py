#!/usr/bin/env python
"""
Main script to generate synthetic EMNIST YOLO dataset
"""

from . import SyntheticDatasetGenerator


def main():
    generator = SyntheticDatasetGenerator()
    generator.generate_dataset()


if __name__ == "__main__":
    main()