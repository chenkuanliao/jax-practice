#!/usr/bin/env python3

from __future__ import annotations

import sys

from llama_block_1gpu import main


if __name__ == "__main__":
    sys.argv[1:1] = [
        "--batch-sequences",
        "1",
        "--seq-len",
        "8192",
        "--json",
        "llama_1gpu_8k.json",
    ]
    main()
