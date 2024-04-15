import os
import pickle
import random

import pandas as pd
import seaborn as sns
import string
from collections import defaultdict, Counter

import networkx as nx
import numpy as np
from nltk.corpus import stopwords
from operator import itemgetter

import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# from MulticoreTSNE import MulticoreTSNE as TSNE

from adjustText import adjust_text
import plotly.graph_objects as go
from sklearn.manifold import TSNE
from wordcloud import WordCloud, STOPWORDS

from PIL import Image

stopwords = set(stopwords.words('english'))
data_folder = "/mnt/nas/coppolillo/WomenHate/"

images_folder = f"{data_folder}/images"

if not os.path.exists(images_folder):
    os.makedirs(images_folder)


def get_emotion(text, tokenizer, model, device):
    input_ids = tokenizer.encode(text[:509] + '</s>', return_tensors='pt')

    output = model.generate(input_ids=input_ids,
                            max_length=2).to(device)

    dec = [tokenizer.decode(ids) for ids in output]
    label = dec[0]
    return label  # "joy", "anger", "hate", "fear", "sadness", "surprise", "love"


import re


def regexp(expr, item):
    reg = re.compile(expr)
    return reg.search(item) is not None


def clean_text(text):
    text = text.lower()
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('[“|”|’|<|>]', '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\d*', '', text)
    text = ' '.join([x for x in text.split() if x not in stopwords])
    return text


def words_tsne_plot(model, word2vec_corpuses, db_names):
    print("Creating TSNE...")

    dbs_colors_dict = {"feminism": "pink", "gendercritical": "pink",
                       "incels": "blue", "mensrights": "blue"}

    labels = []
    tokens = []

    words_colors_dict = defaultdict()

    for i, word2vec_corpus in enumerate(word2vec_corpuses):

        for word, _ in word2vec_corpus:

            color = dbs_colors_dict[db_names[i]]

            if word not in words_colors_dict:
                words_colors_dict[word] = color
                if word not in model:
                    tokens.append(np.random.rand(300))
                else:
                    tokens.append(model[word])
                labels.append(word)
            else:
                if words_colors_dict[word] == "pink" and color == "blue" \
                        or words_colors_dict[word] == "blue" and color == "pink":
                    words_colors_dict[word] = "purple"

    tokens = list(tokens)
    labels = list(labels)

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    plt.figure(figsize=(16, 16))

    texts = []
    for i, (value_x, value_y) in enumerate(new_values):
        color = words_colors_dict[labels[i]]
        zorder = 1
        if color == "purple":
            # plt.annotate(labels[i],
            #              xy=(value_x, value_y),
            #              xytext=(5, 2),
            #              textcoords='offset points',
            #              ha='right',
            #              va='bottom',
            #              zorder=100)
            zorder = 100
            texts.append(plt.text(value_x, value_y, labels[i]))

        plt.scatter(value_x, value_y, s=40, c=color, zorder=zorder)

    adjust_text(texts, only_move={'points': 'y', 'texts': 'xy'})

    tsne_fn = "words_tsne.png"
    tsne_path = os.path.join(images_folder, tsne_fn)

    plt.savefig(tsne_path)

    plt.show()


def language_barplot(word_counters, db_names):

    top_k_common = 20  #  1000

    words_dict = defaultdict(dict)

    for i, word_counter in enumerate(word_counters):
        most_common = word_counter.most_common(top_k_common)
        words_dict[db_names[i]] = most_common

    words = [x[0] for y in words_dict.values() for x in y]
    words = list(set(words))

    df_dict = {}
    for db in words_dict:
        df_words_count = np.zeros(len(words))
        for i, word in enumerate(words):
            d = dict(words_dict[db])
            if word in d:
                df_words_count[i] = d[word]
        df_dict[db] = df_words_count.copy()

    df = pd.DataFrame.from_dict(df_dict, orient='index', columns=words).transpose()
    scaled_df = df/df.sum(0)

    order = scaled_df.sum(axis=1)
    columns_to_order = np.array(words)[np.argsort(order)[::-1]]

    plt.figure()

    scaled_df = scaled_df.reindex(columns_to_order)
    # ax = \
    ax = scaled_df.plot(kind="bar", stacked=True, width=0.8)

    for c in ax.containers:
        # Create a new list of labels
        labels = [round(a*100) if a else "" for a in c.datavalues]
        ax.bar_label(c, labels=labels, label_type="center", rotation=90) # 'edge'

    ax.set_yticklabels(list((ax.get_yticks()*100).astype(int)))
    plot_fn = "language_barplot.png"
    plot_path = os.path.join(images_folder, plot_fn)

    plt.xlabel(f"Most {top_k_common}-Common Words")
    plt.ylabel("%")

    plt.tight_layout()

    plt.savefig(plot_path)

    plt.show()


def get_paths(db_name):
    folder = os.path.join(data_folder, db_name)

    if not os.path.exists(folder):
        os.makedirs(folder)

    emotions_dict_fn = f"emotions_dict.pkl"
    emotions_dict_path = os.path.join(folder, emotions_dict_fn)

    emotions_counter_fn = f"emotions_counter.pkl"
    emotions_counter_path = os.path.join(folder, emotions_counter_fn)

    words_counter_fn = f"words_counter.pkl"
    words_counter_path = os.path.join(folder, words_counter_fn)

    word2vec_corpus_fn = f"word2vec_corpus.npy"
    word2vec_corpus_path = os.path.join(folder, word2vec_corpus_fn)

    return emotions_dict_path, emotions_counter_path, words_counter_path, word2vec_corpus_path


def get_files(db_name):
    paths = list(get_paths(db_name))

    files = []

    for path in paths:
        if path.split(".")[-1] == "pkl":
            with open(path, "rb") as f:
                files.append(pickle.load(f))

        elif path.split(".")[-1] == "npy":
            files.append(np.load(path, allow_pickle=True))

    return files


def get_db_names():
    return ["feminism", "gendercritical", "incels", "mensrights"]


def get_users_graph(db_name):
    users_graph_fn = os.path.join(data_folder, db_name, "graphs", "users_graph.pkl")

    with open(users_graph_fn, "rb") as f:
        users_graph = pickle.load(f)

    return users_graph


def get_users_emotions(db_name):
    users_emotions_fn = os.path.join(data_folder, db_name, "users_emotions.pkl")

    with open(users_emotions_fn, "rb") as f:
        users_emotions = pickle.load(f)

    return users_emotions


emotions_colors_dict = {"neutral": "gray", "hate": "black", "anger": "red",
                        #"joy": "pink",
                        "fear": "lightblue", "sadness": "blue"}
                        # "surprise": "green", "love": "yellow"}


def users_graph_plot(db_name):
    users_graph_fn = os.path.join(data_folder, db_name, "graphs", "users_graph.pkl")

    with open(users_graph_fn, "rb") as f:
        users_graph = pickle.load(f)

    G = nx.DiGraph()

    for k, v in users_graph.items():
        G.add_edges_from(([(k, t) for t in v]))

    SAMPLE_SIZE = 100000

    random_sample_edges = random.sample(list(G.edges), SAMPLE_SIZE)
    G_sample = nx.DiGraph()
    G_sample.add_edges_from(random_sample_edges)

    nx.write_gexf(G_sample, os.path.join(data_folder, db_name, "graphs", f"{db_name}_sample_users_graph.gexf"))


def get_toxicity_scores(db_name):
    toxicity_fn = "toxicity.npy"
    toxicity_path = os.path.join(data_folder, db_name, toxicity_fn)

    toxicity_scores = np.load(toxicity_path)

    return toxicity_scores



def toxicity_plot(toxicity_scores):
    plt.figure(figsize=(10, 10))

    df = pd.DataFrame(columns=["toxicity", "subreddit"])
    df["toxicity"] = np.concatenate(toxicity_scores, axis=0)

    db_names = get_db_names()
    df["subreddit"] = np.concatenate(
        [np.repeat(db_names[i], len(toxicity_scores[i])) for i in range(len(toxicity_scores))], axis=0)

    g = sns.FacetGrid(df, hue='subreddit', height=5)

    # g.map(sns.histplot, 'toxicity', stat="percent", kde=True)
    g.map(sns.kdeplot, 'toxicity', fill=True)
    plt.legend()

    toxicity_image_path = os.path.join(images_folder, "toxicity.png")

    plt.savefig(toxicity_image_path, format="png", dpi=200)
    plt.show()


def get_texts_emotions(db_name):
    emotions_dict_path, emotions_counter_path, \
        words_counter_path, word2vec_corpus_path = get_paths(db_name)

    with open(emotions_dict_path, "rb") as f:
        emotions_dict = pickle.load(f)

    with open(emotions_counter_path, "rb") as f:
        emotions_counter = pickle.load(f)

    return emotions_dict, emotions_counter



def texts_emotions_plot(db_name, texts_emotions_counter):

    sorted_counter = {k: v for k, v in sorted(texts_emotions_counter.items(), key=lambda item: item[1], reverse=True)}
    values = np.array(list(sorted_counter.values()))
    perc = values * 100 / values.sum()

    plt.figure()
    sns.barplot(x=list(sorted_counter.keys()), y=perc, palette=list(emotions_colors_dict.values()))

    emotions_path = os.path.join(images_folder, f"{db_name}_texts_emotions.png")
    plt.savefig(emotions_path, dpi=200)

    plt.show()


def texts_emotions_radar_chart(db_names, text_emotions_counters):

    categories = list(emotions_colors_dict.keys())
    categories.remove("neutral")

    fig = go.Figure()

    max_value = 0

    for i in range(len(db_names)):

        values = np.array(list(itemgetter(*categories)(text_emotions_counters[i])))
        perc = values * 100 / values.sum()
        max_perc = perc.max()

        if max_perc > max_value:
            max_value = max_perc

        fig.add_trace(go.Scatterpolar(
              r=perc,
              theta=categories,
              fill='toself',
              name=db_names[i]
        ))

    fig.update_layout(
      polar=dict(
        radialaxis=dict(
          visible=True,
          range=[0, max_value]
        )),
      showlegend=True
    )

    plt.tight_layout()

    fig.write_image(os.path.join(images_folder, "texts_emotions_radar_chart.png"))
    fig.write_image(os.path.join(images_folder, "texts_emotions_radar_chart.pdf"))

    fig.show()


def users_emotions_radar_chart(db_names, users_emotions_dicts):

    categories = list(emotions_colors_dict.keys())
    categories.remove("neutral")

    fig = go.Figure()

    max_value = 0

    for i in range(len(db_names)):
        D = users_emotions_dicts[i]
        users_emotions_count = np.array([sum(map((c).__eq__, D.values())) for c in categories])
        n_users = len(list(D.keys()))
        perc = users_emotions_count * 100 / n_users

        max_perc = max(perc)

        if max_perc > max_value:
            max_value = max_perc

        fig.add_trace(go.Scatterpolar(
            r=perc,
            theta=categories,
            fill='toself',
            name=db_names[i]
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max_value]
            )),
        showlegend=True
    )

    plt.tight_layout()

    fig.write_image(os.path.join(images_folder, "users_emotions_radar_chart.png"))
    fig.write_image(os.path.join(images_folder, "users_emotions_radar_chart.pdf"))
    fig.show()


def wordscloud(words_counters, db_names):

    mask = np.array(Image.open(r'ciao.png'))

    wordcloud = WordCloud(background_color="rgba(255, 255, 255, 0)", mode="RGBA",
                          stopwords=STOPWORDS,
                          colormap="Reds", mask=mask, random_state=41,
                          collocations=False,
                          width=mask.shape[1], height=mask.shape[0]
                          )

    for i, words_counter in enumerate(words_counters):
        wordcloud.generate_from_frequencies(frequencies=words_counter)
        plt.figure()
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.savefig(os.path.join(images_folder, f"wordscloud_{db_names[i]}.pdf"), dpi=200)
        plt.show()