TEST_FRACTION: float = 0.2
TEST_BEGIN: bool = False

if __name__ == "__main__":
    with open("shakespeare.txt", "r") as f:
        text = f.read()

    n_test = int(len(text) * TEST_FRACTION)

    if TEST_BEGIN:
        test_text = text[:n_test]
        train_text = text[n_test:]
    else:
        test_text = text[-n_test:]
        train_text = text[:-n_test]

    assert (
        abs(len(test_text) / (len(train_text) + len(test_text)) - TEST_FRACTION) < 0.01
    )

    with open("train_shakespeare.txt", "w") as f:
        f.write(train_text)

    with open("test_shakespeare.txt", "w") as f:
        f.write(test_text)
