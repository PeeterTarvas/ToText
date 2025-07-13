#!/bin/sh

SOURCE_DIR="/home/peeter/ToText/synthetic_emnist_yolo"
OUTPUT_ZIP="/home/peeter/ToText/synthetic_emnist_yolo.zip"

cd "$(dirname "$SOURCE_DIR")" || exit 1

echo "Zipping $SOURCE_DIR into $OUTPUT_ZIP..."
zip -r -9 "$OUTPUT_ZIP" "$(basename "$SOURCE_DIR")"

echo "Done!"
