import random

from rasp import full, indices, kqv, seq_map, tok_map

M = 10


def causal_match3_unique_true(x):
    n = len(x)
    i = n - 1
    for j in range(n - 2):
        for k in range(j + 1, n - 1):
            if (x[i] + x[j] + x[k]) % M == 0:
                return [1] * n

    return [0] * n


def equals(a, b):
    return a == b


def true(a, b):
    return True


def causal_match3_unique(x):
    idxs = indices(x)
    last_idx = kqv(k=x, q=x, v=idxs, pred=true, reduction="max")

    last_x = kqv(k=idxs, q=last_idx, v=x, pred=equals, reduction="mean")

    y = tok_map(x, lambda a: (M - a) % M)
    z = seq_map(x, last_x, lambda a, b: (a + b) % M)

    y_idxs = seq_map(y, idxs, lambda a, b: a + b * M)
    z_idxs = seq_map(z, idxs, lambda a, b: a + b * M)

    last_idx_mask = seq_map(
        kqv(k=last_idx, q=idxs, v=full(x, -2), pred=equals, reduction="mean"),
        full(x, 1),
        lambda a, b: a + b,
    )

    y_idxs_masked = seq_map(y_idxs, last_idx_mask, lambda a, b: a * b)
    z_idxs_masked = seq_map(z_idxs, last_idx_mask, lambda a, b: a * b)

    unreduced = kqv(
        k=y_idxs_masked,
        q=z_idxs_masked,
        v=full(x, 1),
        pred=lambda k, q: k >= 0 and q >= 0 and (k % M == q % M) and (k // M != q // M),
        reduction="max",
    )

    reduced = kqv(k=unreduced, q=full(x, 1), v=full(x, 1), pred=equals, reduction="max")

    return reduced.tolist()


if __name__ == "__main__":
    for _ in range(1000):
        x = [random.randint(0, 1000) for _ in range(random.randint(1, 100))]
        assert causal_match3_unique(x) == causal_match3_unique_true(x)
