Project 1: Prokudin-Gorskii Image Alignment

This project provides tools for aligning and color-correcting historical Prokudin-Gorskii
glass plate digitals.

Features:
- Align three vertically stacked RGB channels automatically
- Alignment methods: exhaustive and multi-scale (image pyramid)
- Automatic logging of alignment shifts and processing time into a CSV file.

Usage:
Run from terminal:
python main.py <image_path> [--method METHOD] [--type TYPE]

Arguments:
<image_path>  Path to input image (three vertically stacked RGB channels)
--method      Alignment method: exhaustive or multiscale (default: multiscale)
--type        Alignment type: color or edge (default: color)

Example:
python main.py "../additional_media/harvestors.tif" --method multiscale --type color
