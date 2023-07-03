"""TBD."""
import tensorflow as tf


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
