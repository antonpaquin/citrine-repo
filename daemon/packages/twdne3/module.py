from typing import Dict

import numpy as np
from PIL import Image

from citrine_daemon import create_function
from citrine_daemon.storage.result import FileHandle


input_validator = {
    'truncation': {
        'required': False,
        'default': 0.5,
        'type': 'float',
        'coerce': float,
    },
    'n': {
        'required': False,
        'excludes': 'latents',
        'default': 1,
        'type': 'integer',
        'coerce': int,
    },
    'latents': {
        'required': False,
        'excludes': 'n',
        'type': 'tensor',
        'tensor': {
            'dtype': 'float32',
            'shape': [None, 512],
        },
    },
}


def process_input(params: Dict) -> Dict[str, np.ndarray]:
    if 'latents' in params:
        latents = params['latents']
    else:
        latents = np.random.randn(params['n'], 512).astype('float32')
    return {
        'truncation': np.asarray(params['truncation']).astype('float32'),
        'latents': latents,
    }


def process_output(results: Dict[str, np.ndarray]) -> Dict:
    images = []
    for img_array in results['images']:
        img_array = img_array.transpose(1, 2, 0)
        img_array = (255 * (img_array.clip(-1, 1) + 1) / 2).astype('uint8')
        img = Image.fromarray(img_array)
        images.append(FileHandle.from_pil_image(img))
    return {
        'images': images,
    }


create_function(
    name='generate',
    model='twdne3',
    process_input=process_input,
    process_output=process_output,
    input_validator=input_validator,
)
