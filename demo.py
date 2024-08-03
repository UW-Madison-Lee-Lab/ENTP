import os
import sys

import torch

from nano_transformer import TransformerConfig, TransformerLMHead
from util import Config, Environment, decode, encode

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("usage: python demo.py config-path.json n1 n2")
        exit(1)

    config = Config(sys.argv[1])
    n1 = int(sys.argv[2])
    n2 = int(sys.argv[3])
    n_digits = max(len(str(n1)), len(str(n2)))

    env = Environment()

    chars = sorted(list(set("0123456789+=$\n")))
    vocab_size = len(chars)
    assert vocab_size == 14

    char2int = {c: i for i, c in enumerate(chars)}
    int2char = {i: c for i, c in enumerate(chars)}

    model_config = TransformerConfig(
        n_positions=config.block_size,
        vocab_size=vocab_size,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
    )

    model = TransformerLMHead(model_config).to(env.device)

    load_path = os.path.join(config.model_dir, config.checkpoint_name)
    checkpoint = torch.load(load_path, weights_only=False)
    model.load_state_dict(checkpoint["model"])

    prompt = f"${n1}+{n2}="
    encoded_prompt = encode(prompt, char2int)[None].to(env.device)

    with env.context:
        output = model.generate(encoded_prompt, max_new_tokens=n_digits + 1)

    decoded_output = decode(output[0], int2char)
    print(decoded_output.replace("$", "").removesuffix("\n"))
