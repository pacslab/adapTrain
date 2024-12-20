import importlib

__all__ = []

try:
    Worker = importlib.import_module(".worker", package=__package__).Worker
    __all__.append("Worker")
except ModuleNotFoundError:
    pass  # Ignore if 'worker.py' is not available in the pod

try:
    Controller = importlib.import_module(".controller", package=__package__).Controller
    __all__.append("Controller")
except ModuleNotFoundError:
    pass  # Ignore if 'controller.py' is not available in the pod