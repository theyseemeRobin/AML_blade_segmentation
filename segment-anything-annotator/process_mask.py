import json
import os
import base64
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw


PATH = "output" 

def process_json_to_png(filename):
    print(f"Processing {filename}")

    with open(filename, "r") as file:
        data = json.load(file)

    image = Image.open(BytesIO(base64.b64decode(data['imageData'])))
    image = image.convert("RGBA")

    img = np.asarray(image)

    mask_img = Image.new("1", (img.shape[1], img.shape[0]), 0)
    for shape in data["shapes"]:
        points = [tuple(pair) for pair in shape["points"]]
        ImageDraw.Draw(mask_img).polygon(points, outline=1, fill=1)

    mask = np.array(mask_img)

    new_image_array = np.empty(img.shape, dtype='uint8')

    new_image_array[:,:,:4] = img[:,:,:4]

    new_image_array[:,:,0] = new_image_array[:,:,0] * mask
    new_image_array[:,:,1] = new_image_array[:,:,1] * mask
    new_image_array[:,:,2] = new_image_array[:,:,2] * mask
    new_image_array[:,:,3] = new_image_array[:,:,3] * mask

    cropped = Image.fromarray(new_image_array, "RGBA")
    cropped.save(filename.replace(".json", ".png"))


def process_directory(path):
    if os.path.isfile(path) and path.endswith(".json"):
        process_json_to_png(path)

    if os.path.isfile(path):
        return

    # else it's a dir
    files = os.listdir(path)
    for file in files:
        process_directory(f"{path}/{file}")


def main():
    process_directory(PATH)
    

if __name__ == "__main__":
    main()
