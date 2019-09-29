import spacy
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import operator
import numpy as np
import pickle
import pandas as pd


class ArticleRecommender:
    def __init__(self):
        """
        Article Recommender class
        """
        # self.nlp = spacy.load("es_core_news_md")
        self.nlp = spacy.blank("es")

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r"(jquery *?\([^)]+\)(\.[\w]*\([^)]+\))?)", " ", text)
        text = re.sub(r"(if [^}]+\})", " ", text)
        text = re.sub(r"(else [^}]+\})", " ", text)
        text = re.sub(r"(function [^}]+\})", " ", text)
        text = re.sub(r"\b(var[^;]+;)", " ", text)
        text = re.sub(r"[^\w]", " ", text)
        return text

    def _preprocess(self, text):
        div = [token.text for token in self.nlp(text)]
        return div

    def articles_process(self, articles):
        """
        Esta implementacion guarda todo en memoria, podria cambiarlo
        para que sea mas eficiente, pero decidi hacerlo asi
        ya que es mas simple y en el dataset hay pocos articulos
        """
        articles_text = articles.get("content")
        names = articles.get("title")
        processed_articles = []
        for article in articles_text:
            cleaned = self.clean_text(article)
            text = " ".join(self._preprocess(cleaned))
            processed_articles.append(text)
        return (processed_articles, names)

    def tfidf_compute(self, processed_articles, path_save):
        transvector = TfidfVectorizer()
        corpus = []
        articles, names = processed_articles

        for article in articles:
            corpus.append(article.strip())

        transvector.fit_transform(corpus)
        tfidf1 = transvector.fit_transform(corpus)
        tfidf_matrix = tfidf1.toarray()
        pickle.dump(transvector, open("{}/recommender.pickle".format(path_save), "wb"))
        df = pd.DataFrame(tfidf_matrix)
        df["articles_names"] = names
        df.to_csv("{}/recommender_tfidf.csv".format(path_save), sep="|")

    def query_process(self, path_recommender, path_tfidf, query, num_articles):
        # TODO: prepare to receive any data type
        df = pd.read_csv(path_tfidf, sep="|")
        transvector = pickle.load(open(path_recommender, "rb"))
        names = df["articles_names"]
        df = df.drop("articles_names", axis=1)
        tfidf_matrix = np.array(df)
        word_freq = nltk.FreqDist(self._preprocess(query))
        # word_freq2 = Counter([token.lemma_ for token in nlp(query)])
        # common_words = word_freq2.most_common(5)
        max_freq = word_freq.most_common()[0][1]
        idf = transvector.idf_
        weighted_query = np.zeros((1, tfidf_matrix.shape[1]))

        for word in word_freq:

            word_index = transvector.vocabulary_.get(word)
            if word_index != None:
                # Previniendo posible sesgo causado por documentos grandes
                # usando doble normalizacion
                weighted_query[0, word_index] = (
                    0.5 + 0.5 * word_freq[word] / float(max_freq)
                ) * idf[transvector.vocabulary_.get(word)]

        similarity = cosine_similarity(tfidf_matrix, weighted_query)

        apps_dictionary = {}

        for i in range(len(names)):
            apps_dictionary.update({names[i]: similarity[i]})
        if apps_dictionary.get(query) is not None:
            apps_dictionary.pop(query)

        sorted_apps_dictionary = sorted(
            apps_dictionary.items(), key=operator.itemgetter(1), reverse=True
        )
        result = [s[0] for s in sorted_apps_dictionary[0:num_articles]]
        return result
