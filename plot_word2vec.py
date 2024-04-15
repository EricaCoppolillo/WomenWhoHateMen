import gensim

from utils import *


def main():
    db_names = get_db_names()

    texts_emotions_counters = []

    words_counters = []

    for db_name in db_names:
        _, texts_emotions_counter = get_texts_emotions(db_name)
        texts_emotions_counters.append(texts_emotions_counter)
        _, _, words_counter, _ = get_files(db_name)
        words_counters.append(words_counter)

    language_barplot(words_counters, db_names)
    wordscloud(words_counters, db_names)
    texts_emotions_radar_chart(db_names, texts_emotions_counters)



if __name__ == '__main__':
    main()
