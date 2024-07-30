import sys

sys.path.append("..")

import torch
from nano_transformer import TransformerConfig, TransformerLMHead

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

N_EMBD = 384
N_LAYER = 6
N_HEAD = 6
BLOCK_SIZE = 128


def load_model(path: str) -> TransformerLMHead:
    config = TransformerConfig(
        n_positions=BLOCK_SIZE,
        vocab_size=14,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_embd=N_EMBD,
    )
    model = TransformerLMHead(config).to(device)
    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    return model


if __name__ == "__main__":
    init1 = load_model("models/test_init1.pt")
    init2 = load_model("models/test_init2.pt")
    train1 = load_model("models/test_train1.pt")
    train2 = load_model("models/test_train2.pt")

    for (n1, p1), (n2, p2) in zip(init1.named_parameters(), init2.named_parameters()):
        assert n1 == n2
        assert torch.all(p1 == p2)

    for (n1, p1), (n2, p2) in zip(train1.named_parameters(), train2.named_parameters()):
        assert n1 == n2
        assert torch.allclose(p1, p2)
