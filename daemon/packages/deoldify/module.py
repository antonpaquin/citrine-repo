from typing import Dict

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
            },
        },
    },
}


def process_input(params: Dict) -> Dict[str, np.ndarray]:
    # deoldify wants dynamic model shapes, which the current colorizer.onnx doesn't have.
    # Apparently ONNX + onnxruntime _does_ support this, but that means I need to figure out how to convert it somehow.

    return {
        'input.1': np.ndarray([0]),  # TODO
    }


def process_output(results: Dict[str, np.ndarray]) -> Dict:
    out = results['608']
    return {}


create_function(
    name='run',
    model='deoldify_colorizer',
    process_input=process_input,
    process_output=process_output,
    input_validator=input_validator,
)
