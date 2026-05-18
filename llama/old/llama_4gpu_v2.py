#!/usr/bin/env python3

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
os.environ.setdefault("NCCL_DEBUG", "WARN")

from llama_block_common import main_for_variant


if __name__ == "__main__":
    main_for_variant(4, "v2", __file__)
