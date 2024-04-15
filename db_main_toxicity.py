import sqlite3
# protobuf 4.25.1

import torch
import evaluate

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

    submission_query = "SELECT title || '.' || selftext FROM submission "
    title_query = ' '.join(
        [f"title REGEXP '.*[ .,?!:]{x}[ .,?!:].*' or title REGEXP '^({x}) .*' or" if x != bag_of_words[-1] else
         f"title REGEXP '.*[ .,?!:]{x}[ .,?!:].*' or title REGEXP '^({x}) .*'" for x in bag_of_words])
    final_query = submission_query + f"where ({title_query})"
    c.execute(final_query)

    comment_query = "SELECT body FROM comment "
    body_query = ' '.join(
        [f"body REGEXP '.*[ .,?!:]{x}[ .,?!:].*' or body REGEXP '^({x}) .*' or" if x != bag_of_words[-1] else
         f"body REGEXP '.*[ .,?!:]{x}[ .,?!:].*' or body REGEXP '^({x}) .*'" for x in bag_of_words])
    final_query = comment_query + f"where ({body_query})"

    c.execute(final_query)

    comments = c.fetchall()
    posts = c.fetchall()

    results = comments + posts
    results = [x[0].replace(".[removed]", "").replace(".[deleted]", "") for x in results]

    toxicity = evaluate.load("toxicity", module_type="measurement")

    print("Computing toxicity scores...")
    toxicity_results = toxicity.compute(predictions=results)
    toxicity_scores = [round(s, 4) for s in toxicity_results["toxicity"]]

    if SAVE:
        data_folder = "/mnt/nas/coppolillo/WomenHate/"
        db_folder = os.path.join(data_folder, db_name)

        toxicity_path = os.path.join(db_folder, "toxicity.npy")
        np.save(toxicity_path, toxicity_scores)


if __name__ == '__main__':

    db_names = get_db_names()[::-1]  # order is important
    for db_name in db_names:
        main(db_name)
