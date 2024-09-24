from tracr.compiler import compiling  # type: ignore
from tracr.rasp import rasp  # type: ignore


def kqv(k, q, v, pred):
    return rasp.Aggregate(rasp.Select(k, q, pred), v)


x = rasp.tokens
idxs = rasp.indices
all_true_selector = rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.TRUE)

n = rasp.Map(lambda a: max(a, 1), rasp.SelectorWidth(all_true_selector))
last_idxs = rasp.Map(lambda a: a - 1, n)
last_x = kqv(k=idxs, q=last_idxs, v=x, pred=rasp.Comparison.EQ)
y = rasp.SequenceMap(lambda a, b: a % b, n - x, n)
z = rasp.SequenceMap(lambda a, b: a % b, x + last_x, n)
row_counts = rasp.SelectorWidth(rasp.Select(y, z, rasp.Comparison.EQ))
count = rasp.Aggregate(all_true_selector, row_counts) * n
count_triplets = rasp.SequenceMap(lambda a, b: a % b, count, n)


if __name__ == "__main__":
    N = 5
    bos = "BOS"
    model = compiling.compile_rasp_to_model(
        count_triplets,
        vocab=set(range(N)),
        max_seq_len=N,
        compiler_bos=bos,
    )

    print(f"{count_triplets([0, 0, 2, 1])=}")
    print(f"{model.apply([bos, 0, 0, 2, 1]).decoded=}")

    def count_params(node):
        if isinstance(node, dict):
            return sum(count_params(child) for child in node.values())

        return len(node.reshape(-1))
