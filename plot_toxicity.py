from utils import *


def main():

    db_names = get_db_names()

    if not isinstance(db_names, list):
        db_names = [db_names]

    toxicity_scores = []

    for db_name in db_names:
        db_toxicity_scores = get_toxicity_scores(db_name)
        toxicity_scores.append(db_toxicity_scores)

    toxicity_plot(toxicity_scores)


if __name__ == '__main__':
    main()