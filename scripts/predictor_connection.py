import numpy as np
from PIL import Image
from io import BytesIO

def extend2RGBA(rgb_values): # change name
    
    images = []
    for index, rgbs in enumerate(rgb_values):
        stream = BytesIO()
        im = Image.fromarray(rgbs).convert("L")
        im.save(stream, "PNG")
        images.append(stream)
    return images


