import random

from rasp import full, indices, kqv, sel_width, select, seq_map, tok_map


def count3_true(x):
    n = len(x)
    m = n + 1
    count = 0
    for i in range(n):
        for j in range(n):
            if (x[i] + x[j] + x[-1]) % m == 0:
                count += 1

    return [count % m] * n


def count3_linear(x):
    n = len(x)
    m = n + 1
    count = 0
    mod_counts = [0] * m
    for i in range(n):
        mod_counts[(x[i] + x[-1]) % m] += 1

    for i in range(n):
        count += mod_counts[(m - x[i]) % m]

    return [count % m] * n


def equals(a, b):
    return a == b


def true(a, b):
    return True


def count3_rasp(x):
    idxs = indices(x)
    last_idx = kqv(k=x, q=x, v=idxs, pred=true, reduction="max")
    last_x = kqv(k=idxs, q=last_idx, v=x, pred=equals, reduction="mean")
    n = tok_map(last_idx, lambda a: a + 1)
    m = tok_map(last_idx, lambda a: a + 2)
    y = seq_map(x, m, lambda a, b: (b - a) % b)
    z = seq_map(seq_map(x, last_x, lambda a, b: a + b), m, lambda a, b: a % b)
    row_counts = sel_width(select(k=y, q=z, pred=equals))
    row_counts_times_n = seq_map(row_counts, n, lambda a, b: a * b)
    count = kqv(
        k=full(x, 1),
        q=full(x, 1),
        v=row_counts_times_n,
        pred=equals,
        reduction="mean",
    )
    mod_count = seq_map(count, m, lambda a, b: a % b)
    return mod_count.tolist()


if __name__ == "__main__":
    for _ in range(1000):
        x = [random.randint(0, 1000) for _ in range(random.randint(3, 200))]
        assert count3_rasp(x) == count3_true(x)
        assert count3_linear(x) == count3_true(x)
