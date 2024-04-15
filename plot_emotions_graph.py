from utils import *


def main():

    db_names = get_db_names()

    if not isinstance(db_names, list):
        db_names = [db_names]

    users_emotions_dicts = []
    for db_name in db_names:
        print(f"DB NAME: {db_name}")
        users_graph_plot(db_name)
        users_emotions = get_users_emotions(db_name)
        users_emotions_dicts.append(users_emotions)

    users_emotions_radar_chart(db_names, users_emotions_dicts)


if __name__ == '__main__':
    main()