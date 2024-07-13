import os
import torch
from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache_Q4, ExLlamaV2Tokenizer

def get_model(base, variant, gpu_split, batch_size, model_params):
    
    model_dir = os.path.join(base, variant)
    config = ExLlamaV2Config()
    config.model_dir = model_dir
    config.prepare()
    config.max_batch_size = batch_size

    params = model_params.get(variant, {})
    for param, value in params.items():
        setattr(config, param, value)

    model = ExLlamaV2(config)
    print(f" -- Loading model: {model_dir}")

    if gpu_split:
        model.load(gpu_split)
        cache = None
    else:
        cache = ExLlamaV2Cache_Q4(model, batch_size=batch_size, lazy=True)
        model.load_autosplit(cache)

    tokenizer = ExLlamaV2Tokenizer(config)

    return model, cache, tokenizer