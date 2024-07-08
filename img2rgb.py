from PIL import Image
def is_rgb(image_path):
    image = Image.open(image_path)
    return image.mode == 'RGB'

# Usage
image_path = "boris1_00000.png"
print(is_rgb(image_path))  # True if RGB, False if grayscale