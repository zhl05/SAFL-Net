"""Model backend integration."""
import importlib
import os


class ModelBackendUnavailable(RuntimeError):
    """The configured model backend cannot be loaded."""


def _backend():
    module_name = os.environ.get("SAFL_BACKEND", "safl_model")
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name != module_name and not module_name.startswith(error.name + "."):
            raise
        raise ModelBackendUnavailable(
            f"Model backend '{module_name}' is not installed. "
            "Configure SAFL_BACKEND with an importable backend module."
        ) from error


def build_model(image_size=256, low_image_size=128):
    """Build the configured model for the requested image dimensions.

    The returned module accepts (image, points, boxes, use_auto_prompt=False)
    and returns (segmentation, class_logits, auxiliary_1, auxiliary_2).
    segmentation contains 'masks' and 'low_res_logits'; class_logits is B x 2.
    """
    return _backend().build_model(image_size=image_size, low_image_size=low_image_size)


def build_optimizer(model):
    """Return (optimizer, scheduler); scheduler may be None.

    When provided, the scheduler steps after each successful optimizer update.
    """
    return _backend().build_optimizer(model)
