from sklearn.model_selection import train_test_split


def count_digits(n1: int, n2: int) -> int:
    return max(len(str(n1)), len(str(n2)))


def count_carrys(n1: int, n2: int) -> int:
    digits1 = [int(d) for d in str(n1)][::-1]
    digits2 = [int(d) for d in str(n2)][::-1]

    digits1 += [0] * max(0, len(digits2) - len(digits1))
    digits2 += [0] * max(0, len(digits1) - len(digits2))

    res = 0
    c = 0
    for d1, d2 in zip(digits1, digits2):
        s = d1 + d2 + c
        c = s // 10
        if c > 0:
            res += 1

    return res


def make_file(inputs: list[tuple[int, int]], outputs: list[int], name: str) -> None:
    text = "".join(f"${i}+{j}={k}$\n" for (i, j), k in zip(inputs, outputs))
    with open(f"{name}.txt", "w") as f:
        f.write(text)


max_digits = 3

inputs = []
outputs = []
digits = []
carrys = []

for i in range(10**max_digits):
    for j in range(10**max_digits):
        inputs.append((i, j))
        outputs.append(i + j)
        digits.append(count_digits(i, j))
        carrys.append(count_carrys(i, j))

(
    train_inputs,
    test_inputs,
    train_outputs,
    test_outputs,
    train_digits,
    test_digits,
    train_carrys,
    test_carrys,
) = train_test_split(
    inputs,
    outputs,
    digits,
    carrys,
    test_size=0.1,
    shuffle=True,
    stratify=[t for t in zip(digits, carrys)],
    random_state=42,
)

for i in range(len(train_inputs)):
    n1, n2 = train_inputs[i]
    assert train_outputs[i] == n1 + n2
    assert train_digits[i] == count_digits(n1, n2)
    assert train_carrys[i] == count_carrys(n1, n2)

for i in range(len(test_inputs)):
    n1, n2 = test_inputs[i]
    assert test_outputs[i] == n1 + n2
    assert test_digits[i] == count_digits(n1, n2)
    assert test_carrys[i] == count_carrys(n1, n2)

n_bad_ratios = 0

for d in set(digits):
    for c in set(carrys):
        if c > d:
            continue
        n_train = sum(d == nd and c == nc for nd, nc in zip(train_digits, train_carrys))
        n_test = sum(d == nd and c == nc for nd, nc in zip(test_digits, test_carrys))
        ratio = n_test / (n_train + n_test)
        n_bad_ratios += ratio < 0.095 or 0.105 < ratio

assert n_bad_ratios <= 2

train_outputs_reversed = [str(n)[::-1] for n in train_outputs]
test_outputs_reversed = [str(n)[::-1] for n in test_outputs]

make_file(train_inputs, train_outputs, "train_plain")
make_file(train_inputs, train_outputs_reversed, "train_reversed")
make_file(test_inputs, test_outputs, "test_plain")
make_file(test_inputs, test_outputs_reversed, "test_reversed")
