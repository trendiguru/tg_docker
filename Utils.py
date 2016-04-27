from PIL import Image
from cStringIO import StringIO
import requests
import numpy as np


def get_cv2_img_array(url):
    response = requests.get(url)
    file = StringIO(response.content)
    img = Image.open(file)
    r, g, b = img.split()
    img = Image.merge("RGB", (b, g, r))
    return np.asarray(img)
