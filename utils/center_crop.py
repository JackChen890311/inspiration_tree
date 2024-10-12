import cv2
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, help="Path to image folder")
args = parser.parse_args()
image_folder = args.path
# "/home/jack/Code/Projects/inspiration_tree/input_concepts/white_paper/v0/"

for image_path in os.listdir(image_folder):
    image = cv2.imread(os.path.join(image_folder + image_path))
    # center crop
    h, w, _ = image.shape
    if h > w:
        image = image[h//2 - w//2:h//2 + w//2, :, :]
    else:
        image = image[:, w//2 - h//2:w//2 + h//2, :]

    # save image
    cv2.imwrite(os.path.join(image_folder + image_path), image)