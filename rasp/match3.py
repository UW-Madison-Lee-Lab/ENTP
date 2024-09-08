import random

from rasp import full, indices, kqv, seq_map, tok_map

M = 99


def causal_match3_true(x):
    n = len(x)
    i = n - 1
    for j in range(n):
        for k in range(n):
            if (x[i] + x[j] + x[k]) % M == 0:
                return [1] * n

    return [0] * n


def equals(a, b):
    return a == b


def true(a, b):
    return True


def causal_match3(x):
    idxs = indices(x)
    last_idx = kqv(k=x, q=x, v=idxs, pred=true, reduction="max")
    last_x = kqv(k=idxs, q=last_idx, v=x, pred=equals, reduction="mean")
    y = tok_map(x, lambda a: (M - a) % M)
    z = seq_map(x, last_x, lambda a, b: (a + b) % M)
    unreduced = kqv(k=y, q=z, v=full(x, 1), pred=equals, reduction="max")
    reduced = kqv(k=unreduced, q=full(x, 1), v=full(x, 1), pred=equals, reduction="max")
    return reduced.tolist()


if __name__ == "__main__":
    for _ in range(1000):
        x = [random.randint(0, 1000) for _ in range(random.randint(1, 100))]
        assert causal_match3(x) == causal_match3_true(x)
