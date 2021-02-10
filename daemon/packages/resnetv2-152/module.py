from typing import Dict

import numpy as np
from PIL import Image

from citrine_daemon import create_function


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


with open('synset.txt', 'r') as in_f:
    classes = [*map(str.strip, in_f.readlines())]


def process_input(params: Dict) -> Dict[str, np.ndarray]:
    if 'image' in params:
        images = [params['image']]
    else:
        images = params['images']
        
    vec_input = []
    for img in images:
        img = Image.fromarray(img)
        cropsize = min(img.height, img.width)
        crop_h = (img.height - cropsize) // 2
        crop_w = (img.width - cropsize) // 2
        img = img.crop((crop_w, crop_h, crop_w + cropsize, crop_h + cropsize))
        img = img.resize((224, 224))
        img_data = np.asarray(img).transpose([2, 0, 1])

        mean_vec = np.array([0.485, 0.456, 0.406])
        stddev_vec = np.array([0.229, 0.224, 0.225])
        norm_img_data = np.zeros(img_data.shape).astype('float32')
        for i in range(img_data.shape[0]):
            norm_img_data[i, :, :] = (img_data[i, :, :] / 255 - mean_vec[i]) / stddev_vec[i]
        vec_input.append(norm_img_data)

    return {
        'data': np.asarray(vec_input),
    }


def process_output(results: Dict[str, np.ndarray]) -> Dict:
    res = []
    for row in results['resnetv27_dense0_fwd']:
        top_classes = []
        for cls_idx in np.argsort(row)[:-6:-1]:
            cls_code, cls_name = classes[cls_idx].split(' ', maxsplit=1)
            score = row[cls_idx]
            top_classes.append({
                'score': score,
                'code': cls_code,
                'class': cls_name,
            })
        res.append(top_classes)
    return {
        'classes': res
    }


create_function(
    name='classify',
    model='resnet',
    process_input=process_input,
    process_output=process_output,
    input_validator=input_validator,
)
