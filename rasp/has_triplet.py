import random  # type: ignore

import numpy as np

from rasp import full, indices, kqv, sel_width, select, tok_map  # type: ignore


def has_triplet_true(x):
    n = len(x)
    for i in range(n):
        for j in range(n):
            if (x[0] + x[i] + x[j]) % 128 == 0:
                return 1

    return 0


def has_triplet_linear(x):
    n = len(x)
    mod_counts = [False] * 128
    for i in range(n):
        mod_counts[-x[i] % 128] = True

    for i in range(n):
        if mod_counts[(x[0] + x[i]) % 128]:
            return 1

    return 0


def equals(a, b):
    return a == b


def true(a, b):
    return True


def has_triplet_rasp(x):
    idxs = indices(x)
    first_x = kqv(k=idxs, q=full(x, 0), v=x, pred=equals, reduction="mean", causal=True)

    # use bitmask to compute mod
    y = -x & 127  # y[i] = -x[i] % 128
    z = (first_x + x) & 127  # z[i] = (x[0] + x[i]) % 128

    # max_count[-1] > 0 if there exists (i, j) such that y[i] == z[j]
    max_count = kqv(
        k=full(x, 1),
        q=full(x, 1),
        v=sel_width(select(k=y, q=z, pred=equals)),
        pred=equals,
        reduction="max",
    )

    return tok_map(max_count, lambda a: min(a, 1))  # return 0 or 1


if __name__ == "__main__":
    for _ in range(100):
        x = np.array([random.randint(0, 1000) for _ in range(random.randint(3, 100))])
        assert has_triplet_rasp(x)[-1] == has_triplet_true(x)
        assert has_triplet_linear(x) == has_triplet_true(x)
