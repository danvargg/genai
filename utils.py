"""Style transfer utilities."""
import tensorflow as tf

from src.losses import get_style_content_loss


def preprocess_image(image: tf.Tensor) -> tf.Tensor:
    """
    Preprocesses an image for VGG19.

    Args:
        image: A tensor of shape (height, width, channels) representing an image.  # TODO: make sure backend is channels_last

    Returns:
        A tensor of shape (height, width, channels) representing the preprocessed image.
    """
    return tf.keras.applications.vgg19.preprocess_input(image)


def gram_matrix(input_tensor: tf.Tensor) -> tf.Tensor:
    """
    Computes the Gram matrix of a tensor.

    Args:
        input_tensor: A tensor of shape (height, width, channels) representing the features of an image.

    Returns:
        A tensor of shape (channels, channels) representing the Gram matrix of the input tensor.
    """
    # TODO: input_tensorT or input_tensor?
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensorT, input_tensor)  # TODO: research einsum method
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations


def get_style_image_features(image: tf.Tensor, vgg: tf.keras.Model, num_style_layers: int = 5) -> list:
    """
    Gets the image features.

    Args:
        image: A tensor of shape (height, width, channels) representing an image.
        vgg: A Keras model that outputs the style and content of an image.
        num_style_layers: An int representing the number of style layers to use.

    Returns:
        A list of tensors representing the style image features.
    """
    style_outputs = vgg(preprocess_image(image))
    return [gram_matrix(style_layer) for style_layer in style_outputs[:num_style_layers]]


def calculate_gradients(
        image, content_targets, style_targets, style_weight, content_weight, with_regularization: bool = False
):
    with tf.GradientTape() as tape:
        style_features = get_style_image_features(image, vgg)
        content_features = get_content_image_features(image, vgg)
        loss = get_style_content_loss(
            style_targets, style_features, content_targets, content_features, style_weight, content_weight
        )
        return tape.gradient(loss, image)


def update_image_with_style(image, content_targets, style_targets, optimizer, style_weight, content_weight,
                            with_regularization: bool = False):
    gradients = calculate_gradients(
        image, content_targets, style_targets, style_weight, content_weight, with_regularization
    )
    optimizer.apply_gradients([(gradients, image)])


def fit_style_transfer(
        input_image, style_image, optimizer, epochs: int = 1, steps_per_epoch: int = 1,
        with_regularization: bool = False, style_weight: float = 0.01
):
    for epoch in range(epochs):
        for step in range(steps_per_epoch):
            update_image_with_style(
                input_image, content_targets, style_targets, optimizer, style_weight, content_weight,
                with_regularization
            )
    return input_image, images
