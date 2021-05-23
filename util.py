import random


def test_train_split(file_path, test_ratio=0.1, validation_ratio=0.2):
    with open(file_path, "r") as fp:
        examples = fp.readlines()

    seed = 120
    random.seed(seed)
    random.shuffle(examples)

    validation_index = -1 * int((validation_ratio+test_ratio)*len(examples))
    test_index = -1 * int(test_ratio*len(examples))
    train_examples = examples[:validation_index]
    validation_examples = examples[validation_index: test_index]
    test_examples = examples[test_index:]

    dir_name = "/".join(file_path.split("/")[:-1])+"/"
    with open(dir_name+"train_split.csv", "w") as fp:
        fp.writelines(train_examples)
    with open(dir_name+"validation_split.csv", "w") as fp:
        fp.writelines(validation_examples)
    with open(dir_name+"test_split.csv", "w") as fp:
        fp.writelines(test_examples)

if __name__ == "__main__":
    file_path = ".data/coleridgeinitiative-show-us-the-data/train.csv"
    test_train_split(file_path)