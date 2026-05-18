#!/usr/bin/env python3

from __future__ import annotations

import sys

from llama_block_1gpu import main


if __name__ == "__main__":
    sys.argv[1:1] = [
        "--batch-sequences",
        "128",
        "--seq-len",
        "1024",
        "--json",
        "llama_1gpu_1kx128.json",
    ]
    main()
