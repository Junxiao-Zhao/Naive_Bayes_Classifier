import re
import json
import numpy as np
import pandas as pd
from collections import defaultdict

E = 0.1


def load_text(file_path: str) -> str:
    """Read contents of txt file

    :param file_path: the file path
    :return: the file content in string
    """

    try:
        with open(file_path, 'r') as f:
            return f.read().strip()
    except Exception as e:
        print(e)
        exit()


class naive_bayes:

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.l_c = None
        self.l_w_c = None

    def train_test_split(self, num_train: int):
        """Split the trainning and test set

        :param num_train: the size of trainning set
        :return: the trainning set and the test set
        """

        text = load_text(self.file_path)  # read text
        bios = re.split(r"\n{2,}", text.lower())
        bios_train = bios[:num_train]
        bios_test = bios[num_train:]

        return bios_train, bios_test

    @staticmethod
    def normalization(bios: list, allowed_words: list = []):
        """Normalize the biographies

        :param bios: a list of biographies
        :param allowed_words: the allowed words; default empty
        :return: a dict of biographies with categories as keys
        """

        biographies = defaultdict(list)
        stop_word_pattern = r"\b" + r"\b|\b".join(
            re.split(r"\s+", load_text(r"./stopwords.txt"))) + r"\b"

        if allowed_words:
            not_allowed = r'\b(?!(?:' + '|'.join(allowed_words) + r')\b)\w+\b'

        for each in bios:
            each_bio = re.split(r"\n+", each, maxsplit=2)
            each_bio[0] = each_bio[0].strip()
            each_bio[1] = each_bio[1].strip()
            each_bio[2] = re.sub(stop_word_pattern, "", each_bio[2])
            if allowed_words:  # remove words not in the allowed_words
                each_bio[2] = re.sub(not_allowed, "", each_bio[2])
            each_bio[2] = re.sub(r"\b\S{1,2}\b", "", each_bio[2])
            each_bio[2] = re.sub(r"[^\s\w\d]", "", each_bio[2])
            each_bio[2] = re.sub(r"\s+", " ", each_bio[2]).strip()
            if allowed_words:  # remove redundency
                each_bio[2] = list(set(re.split(" ", each_bio[2])))
            biographies[each_bio[1]].append(each_bio)

        return biographies

    @staticmethod
    def counting(biographies: dict):
        """Calculate Occ_T(C) and Occ_T(W|C)

        :param biographies: a dict of normalized biographies
        :return: Occ_T(C) and Occ_T(W|C)
        """

        occt_c = dict(zip(biographies.keys(), map(len, biographies.values())))
        occt_w_c = defaultdict(lambda: defaultdict(set))

        for categ, bios in biographies.items():

            for i, (_, _, bio) in enumerate(bios):
                words = re.split(r"\s+", bio)

                for w in words:
                    occt_w_c[w][categ].add(i)

        for w in occt_w_c.keys():
            for categ in occt_w_c[w].keys():
                occt_w_c[w][categ] = len(occt_w_c[w][categ])

        occt_w_c = pd.DataFrame.from_dict(occt_w_c, 'index')
        occt_w_c.fillna(0, inplace=True)

        return occt_c, occt_w_c

    @staticmethod
    def probabilities(occt_c: dict, occt_w_c: pd.DataFrame):
        """Calculate L(C) and L(W|C)

        :param occt_c: Occ_T(C)
        :param occt_w_c: Occ_T(W|C)
        :return: L(C) and L(W|C)
        """

        num = sum(occt_c.values())  # the size of current set

        # Freq_T(C) and Freq_T(W|C)
        freqt_c = dict([[k, v / num] for k, v in occt_c.items()])
        freqt_w_c = occt_w_c.apply(lambda col: col / occt_c[col.name])

        # P(C) and P(W|C)
        prob_c = dict([[k, (v + E) / (1 + len(occt_c) * E)]
                       for k, v in freqt_c.items()])
        prob_w_c = freqt_w_c.apply(lambda col: (col + E) / (1 + 2 * E))

        # L(C) and L(W|C)
        l_c = dict([[k, -np.log2(v)] for k, v in prob_c.items()])
        l_w_c = prob_w_c.apply(lambda col: -np.log2(col))

        return l_c, l_w_c

    def train(self, trainning_set: list):
        bios_train = self.normalization(trainning_set)
        occt_c_train, occt_w_c_train = nb.counting(bios_train)
        self.l_c, self.l_w_c = self.probabilities(occt_c_train, occt_w_c_train)

    def predict(self, test_set: list):
        bios_test = self.normalization(test_set, list(self.l_w_c.index))

        l_c_b = defaultdict(dict)

        for categ, bios in bios_test.items():
            for name, _, words in bios:
                l_c_b[categ][name] = self.l_c[categ] + sum(
                    [self.l_w_c.loc[w, categ] for w in words])


if __name__ == "__main__":
    nb = naive_bayes(r"./tinyCorpus.txt")
    bios_train, bios_test = nb.train_test_split(5)
    """ train_bios = nb.normalization(bios_train)
    occt_c_train, occt_w_c_train = nb.counting(train_bios)
    # print(occt_c_train)
    # occt_w_c_train.to_csv("./OCCT(W,C)_train.csv")
    freqt_c, freqt_c_w = nb.probabilities(occt_c_train, occt_w_c_train)
    print(freqt_c)
    freqt_c_w.to_csv("./-Log(W,C)_train.csv") """
    nb.train(bios_train)
    """ bios_test = nb.predict(bios_test)
    with open("test_norm.json", 'w') as f:
        json.dump(bios_test, f) """
