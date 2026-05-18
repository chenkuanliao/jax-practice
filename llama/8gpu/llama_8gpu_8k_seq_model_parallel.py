#!/usr/bin/env python3

from __future__ import annotations

import sys

from llama_block_8gpu import main


if __name__ == "__main__":
    sys.argv[1:1] = [
        "--strategy",
        "seq_model_parallel",
        "--batch-sequences",
        "1",
        "--seq-len",
        "8192",
        "--json",
        "llama_8gpu_8k_seq_model_parallel.json",
    ]
    main()
