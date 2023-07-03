"""TBD."""
import tensorflow as tf

# close to input, low levels define the style -> texture
# high levels, deep in network, define the content -> bird


content_layers = ['block5_conv2']

style_layers = [
    'block1_conv1',
    'block2_conv1',
    'block3_conv1',
    'block4_conv1',
    'block5_conv1',
]

layer_names = content_layers + style_layers


def preprocess_image(image: tf.Tensor) -> tf.Tensor:
    """
    Preprocesses an image for VGG19.

    Args:
        image: A tensor of shape (height, width, channels) representing an image.  # TODO: make sure backend is channels_last

    Returns:
        A tensor of shape (height, width, channels) representing the preprocessed image.
    """
    return tf.keras.applications.vgg19.preprocess_input(image)


def vgg_model(layer_names: list) -> tf.keras.Model:
    """
    Creates a VGG19 model that outputs the style and content of an image.

    Args:
        layer_names: A list of strings representing the names of the layers to use for style and content.

    Returns:
        A Keras model that outputs the style and content of an image.
    """
    vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]

    return tf.keras.Model([vgg.input], outputs)
