import re
import json
import pandas as pd
from collections import defaultdict


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


def remove_stop_word(text: str) -> dict:
    """Return the biographies from the text after normalization

    :param text: a string of text
    :return: a dictionary of biographies with categories as keys
    """

    bios = re.split(r"\n{2,}", text.lower())
    stop_word_pattern = r"\b" + r"\b|\b".join(
        re.split(r"\s+", load_text(r"./stopwords.txt"))) + r"\b"
    biographies = defaultdict(list)

    for each in bios:
        each_bio = re.split(r"\n+", each, maxsplit=2)
        each_bio[0] = each_bio[0].strip()
        each_bio[1] = each_bio[1].strip()
        each_bio[2] = re.sub(stop_word_pattern, "", each_bio[2])
        each_bio[2] = re.sub(r"\b\S{1,2}\b", "", each_bio[2])
        each_bio[2] = re.sub(r"[^\s\w\d]", "", each_bio[2])
        each_bio[2] = re.sub(r"\s+", " ", each_bio[2]).strip()
        biographies[each_bio[1]].append(each_bio)

    return biographies


def counting(biographies: dict):

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

    return occt_c, occt_w_c


if __name__ == "__main__":
    a = remove_stop_word(load_text(r"./bioCorpus.txt"))
    """ with open("./bios.json", 'w') as f:
        json.dump(a, f) """
    # print(a)
    a = counting(a)[1]
    """ for k in a.keys():
        a[k] = dict(a[k])
        for key in a[k].keys():
            a[k][key] = list(a[k][key]) """
    """ with open("./OCCT(W,C).json", 'w') as f:
        json.dump(dict(a), f) """

    df = pd.DataFrame.from_dict(a, "index")
    df.fillna(0, inplace=True)
    df.to_csv("./OCCT(W,C).csv")
