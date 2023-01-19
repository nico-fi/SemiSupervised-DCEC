"""
This script creates a csv file containing information about the fashion mnist dataset.
"""

import glob
from pathlib import Path
from pandas import DataFrame
from PIL import Image, ImageStat


def main():
    """
    Creates a csv file containing information about the fashion mnist dataset.
    """
    input_folder_path = Path("data/raw/fashion_mnist")

    print("Creating csv file...")
    img_format, height, width, mean, std, label = [], [], [], [], [], []
    for image_path in glob.glob(str(input_folder_path / "*.png")):
        img = Image.open(image_path)
        img_format.append(img.format)
        height.append(img.height)
        width.append(img.width)
        stat = ImageStat.Stat(img)
        mean.append(sum(stat.mean) / len(stat.mean))
        std.append(sum(stat.stddev) / len(stat.stddev))
        label.append(int(image_path.split('/')[-1].split('_')[1].split('.')[0]))

    dataframe = DataFrame(
        {
            "format": img_format,
            "height": height,
            "width": width,
            "mean": mean,
            "std": std,
            "label": label
        }
    )
    dataframe.to_csv("data/ge/fashion_mnist.csv", index=False)


if __name__ == "__main__":
    main()
