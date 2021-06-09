import random
if __name__ == "__main__":
    file_path = ".data/train.csv"
    train_path, dev_path, test_path = ".data/train_split.csv", ".data/validation_split.csv", ".data/test_split.csv"
    with open(file_path, 'r') as inp:
        examples = inp.readlines()
    shuffle = True
    if shuffle:
        seed = 13370
        random.seed(seed)
        random.shuffle(examples)

    dev_ratio = 0.20
    test_ratio = 0.10
    dev_index = -1 * int((dev_ratio+test_ratio)*len(examples))
    test_index = -1 * int(test_ratio*len(examples))

    train_examples = examples[:dev_index] 
    dev_examples = examples[dev_index: test_index]
    test_examples = examples[test_index:]

    with open(train_path, 'w') as fp:
        fp.writelines(train_examples)

    with open(dev_path, 'w') as fp:
        fp.writelines(dev_examples)

    with open(test_path, 'w') as fp:
        fp.writelines(test_examples)
