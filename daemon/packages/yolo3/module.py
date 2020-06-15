from typing import *

import numpy as np
from PIL import Image

from hivemind_daemon import create_function


input_validator = {
    'image': {
        'required': True,
        'excludes': 'images',
        'type': 'tensor',
        'tensor': {
            'shape': [None, None, 3],
            'dtype': 'uint8',
        },
    },
    'images': {
        'required': True,
        'excludes': 'image',
        'type': 'list',
        'schema': {
            'type': 'tensor',
            'tensor': {
                'shape': [None, None, 3],
                'dtype': 'uint8',
            }
        },
    }
}


with open('classes.txt', 'r') as in_f:
    classes = [line.strip() for line in in_f.readlines()]


# this function is from yolo3.utils.letterbox_image
def letterbox_image(image, size):
    """resize image with unchanged aspect ratio using padding"""
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image


def preprocess(img):
    model_image_size = (416, 416)
    boxed_image = letterbox_image(img, tuple(reversed(model_image_size)))
    image_data = np.array(boxed_image, dtype='float32')
    image_data /= 255.
    image_data = np.transpose(image_data, [2, 0, 1])
    return image_data


def process_input(params: Dict) -> Dict[str, np.ndarray]:
    if 'image' in params:
        images = [params['image']]
    else:
        images = params['images']
        
    image_inputs = []
    shape_inputs = []
    for image in images:
        image = Image.fromarray(image)
        image_inputs.append(preprocess(image))
        image_size = np.array([image.size[1], image.size[0]], dtype=np.int32)
        shape_inputs.append(image_size)

    return {
        'input_1': np.asarray(image_inputs),
        'image_shape': np.asarray(shape_inputs),
    }


def process_output(results: Dict[str, np.ndarray]) -> Dict:
    boxes = results['yolonms_layer_1/ExpandDims_1:0']
    scores = results['yolonms_layer_1/ExpandDims_3:0']
    selections = results['yolonms_layer_1/concat_2:0']
    
    res = []
    for _, cls, idx in selections:
        y1, x1, y2, x2 = boxes[0, idx]
        score = scores[0, cls, idx]
        class_name = classes[cls]
        res.append({
            'box': [x1, y1, x2, y2],
            'score': score,
            'class': class_name,
        })
    return {
        'results': res,
    }


create_function(
    name='run',
    model='yolo',
    process_input=process_input,
    process_output=process_output,
    input_validator=input_validator,
)
