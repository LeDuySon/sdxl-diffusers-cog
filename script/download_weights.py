#!/usr/bin/env python

import os
import torch
import sys

# append project directory to path so predict.py can be imported
sys.path.append('.')

from dotenv import load_dotenv

from huggingface_hub import snapshot_download
from predict import MODEL_ID, SDXL_MODEL_CACHE

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

snapshot_download(repo_id=MODEL_ID,
                  cache_dir=SDXL_MODEL_CACHE,
                  token=hf_token)
