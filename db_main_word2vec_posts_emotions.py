import os
import pickle
import sqlite3
from collections import defaultdict
from empath import Empath
from collections import Counter
import numpy as np

from nltk import word_tokenize

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelWithLMHead

from utils import *


def main(db_name):
    SAVE = True

    # db_name = "mensrights"  # "incels"  # "gendercritical"  # "feminism"

    print(f"DB {db_name}")

    feminist_subreddits = ["feminism", "trufemcels", "gendercritical"]
    masculine_subreddits = ["incels", "mensrights"]

    if db_name in feminist_subreddits:
        bag_of_words = ["men", "boyfriend", "husband", "man", "boy"]
    elif db_name in masculine_subreddits:
        bag_of_words = ["women", "woman", "girl", "girlfriend", "wife"]

    data_folder = "/mnt/nas/coppolillo/WomenHate/"
    db_folder = os.path.join(data_folder, "datasets")

    db_path = os.path.join(db_folder, db_name + ".db")

    conn = sqlite3.connect(db_path)
    conn.create_function("REGEXP", 2, regexp)

    c = conn.cursor()

    c.execute('''
                  CREATE INDEX IF NOT EXISTS author_idx ON comment (author)
                  ''')
    conn.commit()

    submission_query = "SELECT reddit_id, title || '.' || selftext FROM submission "

    title_query = ' '.join(
        [f"title REGEXP '.*[ .,?!:]{x}[ .,?!:].*' or title REGEXP '^({x}) .*' or" if x != bag_of_words[-1] else
         f"title REGEXP '.*[ .,?!:]{x}[ .,?!:].*' or title REGEXP '^({x}) .*'" for x in bag_of_words])
    final_query = submission_query + f"where ({title_query})"
    c.execute(final_query)

    comment_query = "SELECT reddit_id, body FROM comment "
    body_query = ' '.join(
        [f"body REGEXP '.*[ .,?!:]{x}[ .,?!:].*' or body REGEXP '^({x}) .*' or" if x != bag_of_words[-1] else
         f"body REGEXP '.*[ .,?!:]{x}[ .,?!:].*' or body REGEXP '^({x}) .*'" for x in bag_of_words])
    final_query = comment_query + f"where ({body_query})"

    c.execute(final_query)

    comments = c.fetchall()
    posts = c.fetchall()

    results = comments + posts

    gpu_id = 1

    # choose GPU if available
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-emotion")

    model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-emotion")

    lexicon = Empath()

    emotions_dict = defaultdict(set)

    words_counter = Counter()
    emotions_counter = Counter()

    # count = 0

    for id, text in tqdm(results, total=len(results)):

        text = text.replace(".[removed]", "").replace(".[deleted]", "")
        cleaned_text = clean_text(text)

        # words
        word_tokens = word_tokenize(cleaned_text)

        # filtered_sentence = [w for w in word_tokens if not w.lower() in stopwords]
        counter = Counter(word_tokens)

        # word2vec_corpus.extend(list(np.array(counter.most_common(5))[:, 0]))

        # most_occur = counter.most_common(k)
        words_counter += counter

        # emotions
        t5_emotion = get_emotion(cleaned_text, tokenizer, model, device).replace("<pad> ", "")

        empath_analysis = lexicon.analyze(cleaned_text, categories=["joy", "anger", "hate",
                                                                    "fear", "sadness", "surprise"], normalize=True)

        if empath_analysis is None:
            continue

        empath_emotions = [x for (x, y) in empath_analysis.items() if y > 0]

        if len(empath_emotions) == 0:
            emotions = {"neutral"}
        else:
            emotions = {t5_emotion}.union(set(empath_emotions))

        emotions_dict[id] = emotions

        emotions_counter += Counter(emotions)

        # if count == 50:
        #     break
        # count += 1

    k = 10
    most_common_words = words_counter.most_common(k)
    most_common_emotions = emotions_counter.most_common(3)

    print("Top-10 words:", most_common_words)
    print("Top-3 emotions:", most_common_emotions)

    print("******")

    community_k = 10000
    word2vec_corpus = words_counter.most_common(community_k)

    if SAVE:
        emotions_dict_path, emotions_counter_path, \
            words_counter_path, word2vec_corpus_path = get_paths(db_name)

        with open(emotions_dict_path, "wb") as f:
            pickle.dump(emotions_dict, f)

        with open(emotions_counter_path, "wb") as f:
            pickle.dump(emotions_counter, f)

        with open(words_counter_path, "wb") as f:
            pickle.dump(words_counter, f)

        np.save(word2vec_corpus_path, word2vec_corpus)


if __name__ == '__main__':

    db_names = get_db_names()  # order is important
    db_names = ["mensrights"]
    for db_name in db_names:
        main(db_name)
