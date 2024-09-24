import random

import numpy as np

from rasp import full, indices, kqv, sel_width, select, tok_map

EOS = -1


def count_triplets(x):
    n = len(x)
    count = 0
    mod_counts = [0] * n
    for i in range(n):
        mod_counts[(n - x[i]) % n] += 1

    for i in range(n):
        count += mod_counts[(x[i] + x[-1]) % n]

    return count % n


def equals(a, b):
    return a == b


def true(a, b):
    return True


def count_triplets_decoder_cot_rasp(x):
    idxs = indices(x)
    n = kqv(k=x, q=full(x, EOS), v=idxs, pred=equals, reduction="min", causal=True)
    n = tok_map(n, lambda a: a if a else -2)
    last_x = kqv(k=idxs, q=n - 1, v=x, pred=equals, reduction="mean")
    seq_len = kqv(k=x, q=x, v=idxs, pred=true, reduction="max", causal=True)

    i = seq_len - n
    j = seq_len - 2 * n
    xi = kqv(k=idxs, q=i, v=x, pred=equals, reduction="max", causal=True)
    xj = kqv(k=idxs, q=j, v=x, pred=equals, reduction="max", causal=True)

    y = (n - xi) % n + 1
    z = (last_x + xj) % n + 1

    y_mask_write = (n <= idxs) & (idxs < 2 * n)
    z_mask_write = (2 * n <= idxs) & (idxs < 3 * n)
    y_mask_read = (n < idxs) & (idxs <= 2 * n)
    z_mask_read = (2 * n < idxs) & (idxs <= 3 * n)

    z_count = sel_width(
        select(k=x * y_mask_read, q=z, pred=lambda a, b: a == b and a != 0, causal=True)
    )

    count = kqv(
        k=z_mask_read,
        q=z_mask_read,
        v=n * x * z_mask_read,
        pred=lambda a, b: a & b,
        reduction="mean",
        causal=True,
    )
    ans = count % n

    ans_mask_write = idxs == 3 * n
    eos_mask_write = idxs > 3 * n

    return (
        y * y_mask_write
        + z_count * z_mask_write
        + ans * ans_mask_write
        + EOS * eos_mask_write
    )


def count_triplets_decoder_cot(x):
    x.append(EOS)
    while True:
        y = count_triplets_decoder_cot_rasp(np.array(x))
        if y[-1] == EOS:
            return x[-1]

        x.append(y[-1])


if __name__ == "__main__":
    for _ in range(100):
        x = [random.randint(0, 1000) for _ in range(random.randint(3, 25))]
        assert count_triplets(x) == count_triplets_decoder_cot(x)
