"""TBD."""

# close to input, low levels define the style -> texture
# high levels, deep in network, define the content -> bird
# TODO: document above


CONTENT_LAYERS = ['block5_conv2']

STYLE_LAYERS = [
    'block1_conv1',
    'block2_conv1',
    'block3_conv1',
    'block4_conv1',
    'block5_conv1',
]

LAYER_NAMES = CONTENT_LAYERS + STYLE_LAYERS
