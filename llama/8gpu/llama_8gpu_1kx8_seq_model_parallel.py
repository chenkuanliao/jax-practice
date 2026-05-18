#!/usr/bin/env python3

from __future__ import annotations

import sys

from llama_block_8gpu import main


if __name__ == "__main__":
    sys.argv[1:1] = [
        "--strategy",
        "seq_model_parallel",
        "--batch-sequences",
        "8",
        "--seq-len",
        "1024",
        "--json",
        "llama_8gpu_1kx8_seq_model_parallel.json",
    ]
    main()
