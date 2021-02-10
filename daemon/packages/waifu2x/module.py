from typing import *

import numpy as np
from PIL import Image

from citrine_daemon import create_function, errors
from citrine_daemon.storage.result import FileHandle


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
        'minlength': 1,
        'schema': {
            'type': 'tensor',
            'tensor': {
                'shape': [None, None, 3],
                'dtype': 'uint8',
            },
        },
    },
}


def process_input(params: Dict) -> Tuple[Dict[str, np.ndarray], Tuple]:
    if 'image' in params:
        image = params['image'][np.newaxis, :, :, :]
    else:
        images = params['images']
        shape = images[0].shape
        for image in images:
            if image.shape != shape:
                raise errors.InvalidInput('All batch inputs should be the same shape')
        image = np.asarray(images)
    
    image = image.transpose([0, 3, 1, 2])
    image = image / 255
    pad_h = image.shape[2] % 2
    pad_w = image.shape[3] % 2
    image = np.pad(image, [(0, 0), (0, 0), (18, 18 + pad_h), (18, 18 + pad_w)], mode='reflect')
    return {
        'image': image,
    }, (pad_h, pad_w)


def process_output_pil(results: Dict[str, np.ndarray], pad: Tuple[int, int]) -> Dict:
    upscale = results['upscale']
    unpad = [slice(None) for _ in range(4)]
    if pad[0]:
        unpad[2] = slice(0, -2)
    if pad[1]:
        unpad[3] = slice(0, -2)
    upscale = upscale[tuple(unpad)]
    
    images = []
    for x in upscale:
        x = x.transpose([1, 2, 0])
        x = (x * 255).astype('uint8')
        x = Image.fromarray(x)
        x = FileHandle.from_pil_image(x)
        images.append(x)
    return {'images': images}


def process_output_tensor(results: Dict[str, np.ndarray], pad: Tuple[int, int]) -> Dict:
    upscale = results['upscale']
    unpad = [slice(None) for _ in range(4)]
    if pad[0]:
        unpad[2] = slice(0, -2)
    if pad[1]:
        unpad[3] = slice(0, -2)
    upscale = upscale[tuple(unpad)]

    images = []
    for x in upscale:
        x = x.transpose([1, 2, 0])
        x = (x * 255).astype('uint8')
        images.append(x)
    return {'images': images}


create_function(
    name='upscale',
    model='noise1_scale2',
    process_input=process_input,
    process_output=process_output_pil,
    input_validator=input_validator,
)


create_function(
    name='upscale_tensor',
    model='noise1_scale2',
    process_input=process_input,
    process_output=process_output_tensor,
    input_validator=input_validator,
)
