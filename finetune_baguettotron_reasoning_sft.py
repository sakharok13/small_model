#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from finetune_reasoning_traces_sft import main


if __name__ == "__main__":
    # Override with --model_name if your Baguettotron checkpoint has a different HF path.
    main(default_model_name="PleIAs/Baguettotron")
