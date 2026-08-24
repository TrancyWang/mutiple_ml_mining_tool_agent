"""Tool package with lazy exports so optional models do not load at startup."""

from importlib import import_module

__all__ = [
    'TextMiningTools',
    'text_mining_tools',
    'MachineLearningTools', 
    'ml_tools',
    'VisualizationTools',
    'viz_tools',
]


def __getattr__(name):
    modules = {
        "TextMiningTools": (".text_mining_tools", "TextMiningTools"),
        "text_mining_tools": (".text_mining_tools", "text_mining_tools"),
        "MachineLearningTools": (".ml_tools", "MachineLearningTools"),
        "ml_tools": (".ml_tools", "ml_tools"),
        "VisualizationTools": (".viz_tools", "VisualizationTools"),
        "viz_tools": (".viz_tools", "viz_tools"),
    }
    if name not in modules:
        raise AttributeError(name)
    module_name, attribute = modules[name]
    return getattr(import_module(module_name, __name__), attribute)
