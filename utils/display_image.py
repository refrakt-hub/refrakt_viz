import os

import matplotlib.pyplot as plt


def display_image(img_path: str) -> None:
    """
    Display an image from the given file path if it exists.

    Args:
        img_path (str): Path to the image file.
    """
    if os.path.exists(img_path):
        img = plt.imread(img_path)
        plt.imshow(img)
        plt.axis("off")
        plt.show()
