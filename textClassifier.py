import re
import sys
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
        """The Naive Bayes Classifer

        :param file_path: the file path
        """

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
        :return: a dict of biographies if it's trainning set (empty
        allowd words), else a lists
        """

        if allowed_words:
            biographies = list()
        else:
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
                biographies.append(each_bio)
            else:
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

    @staticmethod
    def actual_prob(row: pd.Series):
        """Recover the actual probabilities

        :param row: L(C_i|B)
        :return: P(C_k|B)
        """

        m = min(row)
        row = row.apply(lambda c: 2**(m - c) if c - m < 7 else 0)
        s = sum(row)

        return row / s

    def train(self, trainning_set: list):
        """Train Naive Bayes Classifer base on the trainning set

        :param trainning_set: a trainning set
        """

        bios_train = self.normalization(trainning_set)
        occt_c_train, occt_w_c_train = nb.counting(bios_train)
        self.l_c, self.l_w_c = self.probabilities(occt_c_train, occt_w_c_train)

    def predict(self, test_set: list):
        """Predict the given test set

        :param test_set: a test set
        :return: P(C|B)
        """

        bios_test = self.normalization(test_set, list(self.l_w_c.index))
        l_c_b = defaultdict(dict)
        labels = list()

        for name, label, words in bios_test:
            labels.append(label)
            for categ in self.l_w_c.columns:
                l_c_b[categ][name] = self.l_c[categ] + sum(
                    [self.l_w_c.loc[w, categ] for w in words])

        pred = pd.DataFrame.from_dict(l_c_b)
        pred = pred.apply(self.actual_prob, axis=1)
        pred.insert(len(pred.columns), "predict",
                    pred.apply(lambda row: row.idxmax(), axis=1))
        pred.insert(len(pred.columns), "actual", labels)

        return pred

    @staticmethod
    def format_output(pred: pd.DataFrame):
        """Generate format output

        :param pred: the prediction dataframe
        :return: a format string
        """

        right = 0
        total = pred.shape[0]
        output = []

        for name, row in pred.iterrows():

            correct = False
            if row.loc["predict"] == row.loc["actual"]:
                right += 1
                correct = True

            output.append("%s.\tPrediction: %s.\t%s.\n" %
                          (name.title(), row.loc["predict"],
                           "Right" if correct else "Wrong"))

            for categ, prob in row.iloc[:-2].items():
                output.append(f"{categ}: {prob:.2f}\t")

            output.append("\n\n")

        acc = right / total
        output.append(f"Overall accuracy: {right} out of {total} = {acc:.2f}.")

        return "".join(output)


args = sys.argv[1:]
nb = naive_bayes(args[0])
bios_train, bios_test = nb.train_test_split(int(args[1]))
nb.train(bios_train)
pred = nb.predict(bios_test)
output = nb.format_output(pred)

with open(args[0].split(".")[0] + "_output.txt", 'w') as f:
    f.write(output)

print(output)
