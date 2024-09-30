import random  # type: ignore

import numpy as np

from rasp import full, indices, kqv, sel_width, select, seq_map  # type: ignore


def count_triplets_true(x):
    n = len(x)
    count = 0
    for i in range(n):
        for j in range(n):
            if (x[i] + x[j] + x[-1]) % n == 0:
                count += 1

    return count % n


def count_triplets_linear(x):
    n = len(x)
    count = 0
    mod_counts = [0] * n
    for i in range(n):
        mod_counts[-x[i] % n] += 1

    for i in range(n):
        count += mod_counts[(x[i] + x[-1]) % n]

    return count % n


def equals(a, b):
    return a == b


def true(a, b):
    return True


def count_triplets_rasp(x):
    idxs = indices(x)
    last_idx = kqv(k=x, q=x, v=idxs, pred=true, reduction="max")
    last_x = kqv(k=idxs, q=last_idx, v=x, pred=equals, reduction="mean")
    n = last_idx + 1
    y = -x % n
    z = (x + last_x) % n
    row_counts = sel_width(select(k=y, q=z, pred=equals))
    count = kqv(
        k=full(x, 1),
        q=full(x, 1),
        v=row_counts * n,
        pred=equals,
        reduction="mean",
    )
    return count % n


def count_triplets_rasp_no_div(x):
    idxs = indices(x)
    n = sel_width(select(k=x, q=x, pred=true))
    last_x = kqv(k=idxs, q=n - 1, v=x, pred=equals, reduction="mean")
    y = seq_map(n - x, n, lambda a, b: a if a < b else a - b)
    z = seq_map(x + last_x, n, lambda a, b: a if a < b else a - b)
    row_counts = sel_width(select(k=y, q=z, pred=equals))
    c = kqv(
        k=full(x, 1),
        q=full(x, 1),
        v=row_counts * n,
        pred=equals,
        reduction="mean",
    )
    c = c - idxs * n
    c = kqv(k=c, q=n, v=c, pred=lambda a, b: 0 <= a and a < b, reduction="mean")
    return c


if __name__ == "__main__":
    x = np.zeros(100, dtype=int)
    assert count_triplets_rasp(x)[-1] == count_triplets_true(x)
    assert count_triplets_rasp_no_div(x)[-1] == count_triplets_true(x)
    assert count_triplets_linear(x) == count_triplets_true(x)

    for _ in range(100):
        n = random.randint(4, 100)
        x = np.array([random.randint(0, n - 1) for _ in range(n)])
        assert count_triplets_rasp(x)[-1] == count_triplets_true(x)
        assert count_triplets_rasp_no_div(x)[-1] == count_triplets_true(x)
        assert count_triplets_linear(x) == count_triplets_true(x)
