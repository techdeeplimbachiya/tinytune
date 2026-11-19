"""TinyTune: fine-tune small transformers with LoRA on a single (free) GPU.

Heavy libraries (torch, transformers, datasets, peft, accelerate) are imported
inside the functions that need them, so lightweight modules such as
``tinytune.metrics`` and ``tinytune.config`` can be used and tested without them.
"""

__version__ = "0.1.0"
