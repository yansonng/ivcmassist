"""Loading the published segmentation models."""

import os

os.environ.setdefault("KERAS_BACKEND", "torch")  # must precede the keras import

import keras  # noqa: E402


def load_segmentation_model(model_path, compile_model: bool = False):
    """Load a published ``.keras`` model for inference.

    ``compile_model`` is off by default, and deliberately so. Checkpoints
    written by the original study code record their loss and metrics as
    ``dc_utils.DiceLoss`` / ``nerves_utils.DiceLoss``; restoring the compile
    configuration therefore tries to import those modules by name and fails
    anywhere outside the original working directory. Inference needs neither
    the optimizer nor the loss, so skipping compilation avoids the problem,
    loads faster, and means no custom objects have to be registered at all.

    The published weights carry no compile configuration -- it is dropped when
    the optimizer state is stripped -- so they load cleanly either way.
    """
    return keras.models.load_model(str(model_path), compile=compile_model)
