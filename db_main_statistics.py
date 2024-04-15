import sqlite3

import pandas as pd

from utils import *


def main(db_main):
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

    submission_query = '''SELECT author, title || '.' || selftext FROM submission where author != '[deleted]' and author != '[removed]' '''
    title_query = ' '.join(
        [f"title REGEXP '.*[ .,?!:]{x}[ .,?!:].*' or title REGEXP '^({x}) .*' or" if x != bag_of_words[-1] else
         f"title REGEXP '.*[ .,?!:]{x}[ .,?!:].*' or title REGEXP '^({x}) .*'" for x in bag_of_words])
    final_query = submission_query + f"and ({title_query})"
    c.execute(final_query)

    submission_results = c.fetchall()

    authors = [x[0] for x in submission_results]

    n_authors = len(set(authors))
    print("#Authors: " + str(n_authors))

    posts = [[x[1]] for x in submission_results]

    authors_posts = dict(zip(authors, posts))

    print("#Posts: " + str(len(posts)))

    comment_query = '''SELECT author, body FROM comment where author != '[deleted]' and author != '[removed]' '''
    body_query = ' '.join(
        [f"body REGEXP '.*[ .,?!:]{x}[ .,?!:].*' or body REGEXP '^({x}) .*' or" if x != bag_of_words[-1] else
         f"body REGEXP '.*[ .,?!:]{x}[ .,?!:].*' or body REGEXP '^({x}) .*'" for x in bag_of_words])
    final_query = comment_query + f"and ({body_query})"

    c.execute(final_query)

    comments_results = c.fetchall()

    commenters = [x[0] for x in comments_results]

    n_commenters = len(set(commenters))

    comments = [x[1] for x in comments_results]
    commenters_comments = dict(zip(commenters, comments))

    users_contents = defaultdict(list, authors_posts)

    for k, values in commenters_comments.items():
        users_contents[k].append(values)

    print("#Commenters: " + str(n_commenters))
    print("#Comments: " + str(len(comments)))

    users = set(authors).union(set(commenters))

    print("#Users: ", len(users))

    if SAVE:
        data_folder = "/mnt/nas/coppolillo/WomenHate/"
        db_folder = os.path.join(data_folder, db_name)

        statistics_path = os.path.join(db_folder, "statistics.csv")
        statistics = [db_name, n_authors, len(posts), n_commenters, len(comments), len(users)]
        df = pd.DataFrame([statistics], columns=["subreddit", "authors", "posts", "commenters", "comments", "users"])
        df.to_csv(statistics_path)

        users_contents_path = os.path.join(db_folder, "users_contents.pkl")
        with open(users_contents_path, "wb") as f:
            pickle.dump(users_contents, f)


if __name__ == '__main__':

    db_names = get_db_names()  # order is important
    for db_name in db_names:
        main(db_name)
