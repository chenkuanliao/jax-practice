#!/usr/bin/env python3

from __future__ import annotations

import sys

from llama_block_4gpu import main


if __name__ == "__main__":
    sys.argv[1:1] = [
        "--strategy",
        "tensor_parallel",
        "--batch-sequences",
        "128",
        "--seq-len",
        "1024",
        "--json",
        "llama_4gpu_1kx128_tensor_parallel.json",
    ]
    main()
