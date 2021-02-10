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


imagenet_mean = np.asarray([0.485, 0.456, 0.406]).reshape([1, 3, 1, 1])
imagenet_stdev = np.asarray([0.229, 0.224, 0.225]).reshape([1, 3, 1, 1])


def get_padding(h, w):
    pad_h = (-1 * h) % 16
    pad_h0 = pad_h // 2
    pad_h1 = pad_h - pad_h0

    pad_w = (-1 * w) % 16
    pad_w0 = pad_w // 2
    pad_w1 = pad_w - pad_w0
    return [(0, 0), (0, 0), (pad_h0, pad_h1), (pad_w0, pad_w1)]


def invert_padding(padding) -> [slice]:
    return (
        slice(None),
        slice(None),
        slice(padding[2][0], -padding[2][1]),
        slice(padding[3][0], -padding[3][1]),
    )


def process_image(im: Image) -> np.ndarray:
    im = im.convert('LA').convert('RGB')
    im = np.asarray(im) / 255
    im = im.astype('float32')
    return im


def process_input_full(params: Dict) -> (Dict[str, np.ndarray], [(int, int)]):
    if 'image' in params:
        im = Image.fromarray(params['image'])
        images = process_image(im)[np.newaxis, :, :, :]
    else:
        images = []
        shape = params['images'][0].shape
        for im in params['images']:
            if im.shape != shape:
                raise errors.InvalidInput('All batch inputs should be the same shape')
            im = Image.fromarray(im)
            images.append(process_image(im))
        images = np.asarray(images)
        
    images = images.transpose([0, 3, 1, 2])
    images = (images - imagenet_mean) / imagenet_stdev
    padding = get_padding(images.shape[2], images.shape[3])
    images = np.pad(images, padding, mode='reflect')
            
    return {
        'image': images,
    }, padding


def process_output_full(results: Dict[str, np.ndarray], padding: [(int, int)]) -> Dict:
    images = results['deoldify']
    unpad = invert_padding(padding)
    images = images[unpad]
    images = (images * imagenet_stdev) + imagenet_mean
    images = (images.clip(0, 1) * 255).astype('uint8')
    images = images.transpose([0, 2, 3, 1])

    results = []
    for image in images:
        im = Image.fromarray(image)
        results.append(FileHandle.from_pil_image(im))
        
    return {
        'images': results,
    }


create_function(
    name='fullres',
    model='deoldify_colorizer',
    process_input=process_input_full,
    process_output=process_output_full,
    input_validator=input_validator,
)


def process_input_chroma(params: Dict) -> (Dict[str, np.ndarray], [Image]):
    chroma_size = (256, 256)  # should be evenly divisible by 16

    if 'image' in params:
        orig = Image.fromarray(params['image'])
        im = orig.resize(chroma_size)
        images = process_image(im)[np.newaxis, :, :, :]
        originals = [orig]
    else:
        images = []
        originals = []
        for im in params['images']:
            orig = Image.fromarray(im)
            im = orig.resize(chroma_size)
            images.append(process_image(im))
            originals.append(orig)
        images = np.asarray(images)

    images = images.transpose([0, 3, 1, 2])
    images = (images - imagenet_mean) / imagenet_stdev

    return {
               'image': images,
           }, originals


def process_output_chroma(results: Dict[str, np.ndarray], originals: [Image]) -> Dict:
    images = results['deoldify']
    images = (images * imagenet_stdev) + imagenet_mean
    images = (images.clip(0, 1) * 255).astype('uint8')
    images = images.transpose([0, 2, 3, 1])

    results = []
    for chrom, lum in zip(images, originals):
        chrom = Image.fromarray(chrom).resize(lum.size)
        chrom = np.asarray(chrom.convert('YCbCr'))
        lum = np.asarray(lum.convert('YCbCr'))
        im = np.stack((lum[:, :, 0], chrom[:, :, 1], chrom[:, :, 2]), axis=2)
        im = Image.fromarray(im, mode='YCbCr')
        im = im.convert('RGB')
        results.append(FileHandle.from_pil_image(im))

    return {
        'images': results,
    }


# Method stolen from the original deoldify repo
# Running deoldify on whole inputs is time-consuming, but the human eye isn't actually all that sensitive to color 
# detail. We can get much faster and acceptably good behavior by cheating.
# 1. scale the input image to a small 256x256 square
# 2. run the deoldify network on the small input (it goes pretty fast)
# 3. scale the output back up to match the input
# 4. paste the chroma information of the colorized version on top of the luminance data of the original

create_function(
    name='chroma',
    model='deoldify_colorizer',
    process_input=process_input_chroma,
    process_output=process_output_chroma,
    input_validator=input_validator,
)



