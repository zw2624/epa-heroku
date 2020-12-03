from bs4 import BeautifulSoup
import pandas as pd
import collections
import re
import numpy as np

import spacy
from spacy.lang.en import English
nlp = English()

spacy_stopwords = list(spacy.lang.en.stop_words.STOP_WORDS)

def wield_characters_replace(s):
    rep = collections.OrderedDict()
    rep['â€œ'] = ' “ '
    rep['â€™'] = ' ’ '
    rep['â€˜'] = ' ‘ '
    rep['â€”'] = ' – '
    rep['â€“'] = ' — '
    rep['â€¢'] = ' - '
    rep['â€¦'] = ' … '
    rep['â€'] = ' ” '
    for i, j in rep.items():
        try:
            s = s.replace(i, j)
        except AttributeError:
            return np.NaN

    return s

def clean_str(string):
    string = wield_characters_replace(string)
    string = BeautifulSoup(string, "lxml").get_text()
    string = re.sub('\n', ' ', string)
    string = re.sub('\r', ' ', string)
    string = re.sub(r"[^A-Za-z0-9(),!?\.\…–—-“”’‘]", " ", string)
    string = re.sub("\s{2,}", " ", string)
    return string.strip().lower()

def remove_na(df):
    # filter na
    df = df[df["answer_text"] == df["answer_text"]] # na != na
    # filter "see below"
    df = df[df["answer_text"] != 'see below']
    #filter "See below"
    df = df[df["answer_text"] != 'See below']
    df = df[df["answer_text"] != 'see above']
    df = df[df["answer_text"] != 'see above comments']
    df = df[df["answer_text"] != 'see general comments at the end.']
    df = df[df["answer_text"] != 'Please follow the Preceptors evaluation']
    df = df[df["answer_text"] != 'Please refer to the Preceptor/ grader\'s evaluation']
    df = df[df["answer_text"] != 'please see Preceptors evaluation']
    df = df[df["answer_text"] != 'same as above']
    df = df[df["answer_text"] != 'None']
    df = df[df["answer_text"] != 'none']
    df = df[df["answer_text"] != '#NAME']
    df = df[df["answer_text"] != 'See comments below']
    df = df[df["answer_text"] != 'Comments not available']
    df = df[df["answer_text"].apply(lambda x: x.isnumeric())==False]
    df = df[df["answer_text"].apply(lambda x: 'see question ' not in x) == True]
    df = df[df["answer_text"].apply(lambda x: 'same as ' not in x)== True]
    return df

def judge_bullet(string):
    string = [i for i in string if i.isalpha()]
    return len(string) <= 1

def parse_sentence(string):
    if 'sentencizer' not in nlp.pipe_names:
        sentence_pip = nlp.create_pipe('sentencizer')
        nlp.add_pipe(sentence_pip)
    doc = nlp(string)
    sents = [i.text for i in doc.sents]
    return [i for i in sents if not judge_bullet(i)]

def remove_stop_words(string):
    stop_words = set(spacy_stopwords)
    doc = nlp(string)
    word_list = [token.text for token in doc]
    word_list = [w for w in word_list if not w in stop_words]
    return nlp(" ".join(word_list))

def text_lemmatization(word):
    return word.lemma_


def text_preprocess(string):
    words = [text_lemmatization(word) for word in remove_stop_words(string)]
    return "".join(words)

def basic_clean(string):
    sentences = parse_sentence(clean_str(string))
    return sentences


def delimited_data(file_name):
    data = pd.read_excel(file_name, encoding='utf-8')
    data['comment_id'] = [i for i in range(len(data))]
    data_use = data[["comment_id", "answer_text", "Student", "eval_end_date"]]
    data_use = remove_na(data_use)
    ans_commentid = []
    ans_sent = []
    ans_raw_sent = []
    ans_location = []
    ans_name = []
    ans_date = []
    for i, row in data_use.iterrows():
        if i == 0:
            print(row)
        try:
            raw = basic_clean(row["answer_text"])
            sent = [text_preprocess(j) for j in raw]
            ans_sent.extend(sent)
            ans_raw_sent.extend(raw)
            ans_commentid.extend([row["comment_id"]] * len(sent))
            ans_name.extend([row["Student"]] * len(sent))
            ans_date.extend([row["eval_end_date"]] * len(sent))
            ans_location.extend([k + 1 for k in range(len(sent))])
        except AttributeError:
            continue
    ans_data = {'comment_id': ans_commentid,
                'student_name': ans_name,
                'eval_date': ans_date,
                'answer_sentence': ans_sent,
                'sentence_location': ans_location,
                'sentence_raw': ans_raw_sent}
    df_ans = pd.DataFrame(ans_data)
    return df_ans




