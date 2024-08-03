VAL_FRACTION: float = 0.2
VAL_BEGIN: bool = False

if __name__ == "__main__":
    with open("shakespeare.txt", "r") as f:
        text = f.read()

    n_test = int(len(text) * VAL_FRACTION)

    if VAL_BEGIN:
        val_text = text[:n_test]
        train_text = text[n_test:]
    else:
        val_text = text[-n_test:]
        train_text = text[:-n_test]

    assert abs(len(val_text) / (len(train_text) + len(val_text)) - VAL_FRACTION) < 0.01

    with open("train_shakespeare.txt", "w") as f:
        f.write(train_text)

    with open("val_shakespeare.txt", "w") as f:
        f.write(val_text)
