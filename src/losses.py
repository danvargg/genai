"""TBD."""
import tensorflow as tf


def get_content_loss(features: tf.Tensor, targets: tf.Tensor, weight: float = 0.5) -> tf.Tensor:
    """
    Gets the content loss between two tensors.

    Args:
        features: A tensor of shape (height, width, channels) representing the features of an image.
        targets: A tensor of shape (height, width, channels) representing the features of an image.
        weight: A float representing the weight of the content loss.

    Returns:
        A tensor representing the content loss between the two tensors.
    """
    return weight * tf.reduce_mean(tf.square(features - targets))


def get_style_loss(features: tf.Tensor, targets: tf.Tensor, weight: float = 0.5) -> tf.Tensor:
    """
    Gets the style loss between two tensors.

    Args:
        features: A tensor of shape (height, width, channels) representing the features of an image.
        targets: A tensor of shape (height, width, channels) representing the features of an image.
        weight: A float representing the weight of the content loss.

    Returns:
        A tensor representing the content loss between the two tensors.
    """
    return weight * tf.reduce_mean(tf.square(features - targets))


def get_style_content_loss(  # TODO: typings and docstring
        style_targets, style_outputs, content_targets, content_outputs, style_weight, content_weight
):
    style_loss = tf.add_n([
        get_style_loss(
            style_output, style_target
        ) for style_output, style_target in zip(style_outputs, style_targets)
    ])

    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n([
        get_content_loss(
            content_output, content_target
        ) for content_output, content_target in zip(content_outputs, content_targets)
    ])

    content_loss *= content_weight / num_content_layers

    return style_loss + content_loss
