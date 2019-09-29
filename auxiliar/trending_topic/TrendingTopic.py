from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from spacy.lang.es.stop_words import STOP_WORDS
from sklearn.decomposition import LatentDirichletAllocation
import pickle
import pandas as pd


class TrendingTopic:
    def __init__(self):
        pass

    def train(self, articles, path_save, n_features, n_topics, n_iter):
        tf_vectorizer = CountVectorizer(
            max_df=0.95, min_df=2, max_features=n_features, stop_words=STOP_WORDS
        )
        tf = tf_vectorizer.fit_transform(articles)
        tf_feature_names = tf_vectorizer.get_feature_names()
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            max_iter=n_iter,
            learning_method="online",
            learning_offset=50.0,
            random_state=0,
        ).fit(tf)
        pickle.dump(lda, open("{}/trending_topic.pickle".format(path_save), "wb"))
        df = pd.DataFrame(tf_feature_names)
        df.to_csv("{}/trending_topic_matrix.csv".format(path_save), sep="|")

    def display(self, model, feature_names, n_top_words):
        df = pd.read_csv(feature_names, sep="|")
        features = df['0'].tolist()
        tt_model = pickle.load(open(model, "rb"))
        for topic_idx, topic in enumerate(tt_model.components_):
            print("Topic {}".format(topic_idx))
            print(
                " ".join(
                    [features[i] for i in topic.argsort()[: -n_top_words - 1 : -1]]
                )
            )

