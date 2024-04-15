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


def main():
    SAVE_GRAPH = True

    print(f"DB {db_name}")

    feminist_subreddits = ["feminism", "trufemcels", "gendercritical"]
    masculine_subreddits = ["incels", "mensrights"]

    if db_name in feminist_subreddits:
        bag_of_words = ["men", "boyfriend", "boyfriends",
                        "husband", "husbands", "man", "boy", "boys", "male", "males"]
    elif db_name in masculine_subreddits:
        bag_of_words = ["women", "woman", "girl", "girls",
                        "girlfriend", "girlfriends", "wife", "wifes", "female", "females"]

    data_folder = "/mnt/nas/coppolillo/WomenHate/"
    db_folder = os.path.join(data_folder, "datasets")

    db_path = os.path.join(db_folder, db_name + ".db")

    gpu_id = 1
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-emotion")

    model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-emotion")

    lexicon = Empath()

    conn = sqlite3.connect(db_path)
    conn.create_function("REGEXP", 2, regexp)

    c = conn.cursor()

    c.execute('''
                  CREATE INDEX IF NOT EXISTS author_idx ON comment (author)
                  ''')
    conn.commit()

    submission_query = '''SELECT author FROM submission where author != '[deleted]' and author != '[removed]' '''
    title_query = ' '.join(
        [f"title REGEXP '.*[ .,?!:]{x}[ .,?!:].*' or title REGEXP '^({x}) .*' or" if x != bag_of_words[-1] else
         f"title REGEXP '.*[ .,?!:]{x}[ .,?!:].*' or title REGEXP '^({x}) .*'" for x in bag_of_words])
    final_query = submission_query + f"and ({title_query})"
    c.execute(final_query)

    authors = [x[0] for x in c.fetchall()]

    comment_query = '''SELECT author FROM comment where author != '[deleted]' and author != '[removed]' '''
    body_query = ' '.join(
        [f"body REGEXP '.*[ .,?!:]{x}[ .,?!:].*' or body REGEXP '^({x}) .*' or" if x != bag_of_words[-1] else
         f"body REGEXP '.*[ .,?!:]{x}[ .,?!:].*' or body REGEXP '^({x}) .*'" for x in bag_of_words])
    final_query = comment_query + f"and ({body_query})"

    c.execute(final_query)

    commenters = [x[0] for x in c.fetchall()]

    users = set(authors).union(set(commenters))

    users_emotions_dict = {}

    # count = 0

    n_users = len(users)
    users_dict = dict(zip(users, range(n_users)))

    for user in tqdm(users, total=len(users)):
        posts_query = f"SELECT title || '.' || selftext FROM submission where author = '{user}'"
        c.execute(posts_query)

        posts = [x[0] for x in c.fetchall()]

        comments_query = f"SELECT body FROM comment where author = '{user}'"

        c.execute(comments_query)

        comments = [x[0] for x in c.fetchall()]

        user_texts = np.concatenate((posts, comments))

        user_emotions = []

        for text in user_texts:
            text = text.replace(".[removed]", "").replace(".[deleted]", "")
            cleaned_text = clean_text(text)

            if cleaned_text == '':
                continue

            t5_emotion = get_emotion(cleaned_text, tokenizer, model, device).replace("<pad> ", "")

            empath_analysis = lexicon.analyze(cleaned_text, categories=["joy", "anger", "hate",
                                                                        "fear", "sadness", "surprise"], normalize=True)

            empath_emotions = [x for (x, y) in empath_analysis.items() if y > 0]

            if len(empath_emotions) == 0:
                emotions = {"neutral"}
            else:
                emotions = {t5_emotion}.union(set(empath_emotions))

            user_emotions.extend(emotions)

        if len(user_emotions) == 0:
            user_emotions = ["neutral"]

        counter = Counter(user_emotions)
        prevalent_emotion = max(counter, key=counter.get)

        users_emotions_dict[users_dict[user]] = prevalent_emotion

        # if count == 100:
        #     break
        #
        # count += 1

    commenters_to_authors = '''SELECT c.author, s.author
         FROM comment as c, submission as s where c.parent_id = s.reddit_id and
          c.author != '[deleted]' and c.author != '[removed]'
         and s.author != '[deleted]' and s.author != '[removed]' '''
    final_query = commenters_to_authors + f"and ({body_query})" + f" and ({title_query})"

    c.execute(final_query)

    results = c.fetchall()

    users_graph = defaultdict(set)

    for result in tqdm(results, total=len(results)):
        commenter, post_author = result

        users_graph[users_dict[commenter]].add(users_dict[post_author])

    comments_to_comment = '''SELECT c1.author, c2.author 
             FROM comment as c1, comment as c2 where c1.parent_id = c2.reddit_id and
              c1.author != '[deleted]' and c1.author != '[removed]'
             and c2.author != '[deleted]' and c2.author != '[removed]' '''
    comment_query = ' '.join(
        [f"c1.body REGEXP '.*[ .,?!:]{x}[ .,?!:].*' or c1.body REGEXP '^({x}) .*' or" if x != bag_of_words[-1]
         else f"c1.body REGEXP '.*[ .,?!:]{x}[ .,?!:].*' or c1.body REGEXP '^({x}) .*'" for x in bag_of_words])
    comment2_query = ' '.join(
        [f"c2.body REGEXP '.*[ .,?!:]{x}[ .,?!:].*' or c2.body REGEXP '^({x}) .*' or" if x != bag_of_words[-1]
         else f"c2.body REGEXP '.*[ .,?!:]{x}[ .,?!:].*' or c2.body REGEXP '^({x}) .*'" for x in bag_of_words])

    final_query = comments_to_comment + f"and ({comment_query})" + f" and ({comment2_query})"

    c.execute(final_query)

    results = c.fetchall()

    for result in tqdm(results, total=len(results)):
        commenter, previous_commenter = result

        users_graph[users_dict[commenter]].add(users_dict[previous_commenter])

    if SAVE_GRAPH:
        graphs_folder = os.path.join(data_folder, db_name, "graphs")

        if not os.path.exists(graphs_folder):
            os.makedirs(graphs_folder)

        users_graph_fn = f"users_graph.pkl"

        users_graph_path = os.path.join(graphs_folder, users_graph_fn)

        with open(users_graph_path, "wb") as f:
            pickle.dump(users_graph, f)

        users_emotions_fn = "users_emotions.pkl"

        users_emotions_path = os.path.join(data_folder, db_name, users_emotions_fn)

        with open(users_emotions_path, "wb") as f:
            pickle.dump(users_emotions_dict, f)


if __name__ == '__main__':

    db_names = get_db_names()  # order is important
    for db_name in db_names:
        main(db_name)
