import pandas as pd
import numpy as np
import os.path
from pprint import pprint
from auxiliar.data_preparation.DataPreparation import DataPreparation
from auxiliar.article_recommender.ArticleRecommender import ArticleRecommender
from auxiliar.trending_topic.TrendingTopic import TrendingTopic

PWD = os.path.dirname(__file__)
CONFIG = os.path.join(PWD, "config.json")


def prepare_data():
    bbva = DataPreparation(CONFIG)
    path_processed = bbva.config.get("path_processed")
    scr = bbva.preprocess_scraping()
    scr.index = range(len(scr.index))
    scr.to_csv(
        "{}/processed_scraping.csv".format(path_processed),
        sep="|",
        index=True,
        index_label="index",
    )
    ana = bbva.preprocess_analytics()
    ana.index = range(len(ana.index))
    ana.to_csv(
        "{}/processed_analytics.csv".format(path_processed),
        sep="|",
        index=True,
        index_label="index",
    )


def article_recommender_train(scraping_csv, path_save):
    articles = pd.read_csv(scraping_csv, sep="|")
    recommender = ArticleRecommender()
    processed_articles = recommender.articles_process(articles)
    recommender.tfidf_compute(processed_articles, path_save)


def article_recommender_query(path_recommender, path_tfidf, query, num_articles):
    recommender = ArticleRecommender()
    recommendations = recommender.query_process(
        path_recommender, path_tfidf, query, num_articles
    )
    print("Quizá podrían interesarte los siguientes articulos: ")
    for rec in recommendations:
        print(" -" + rec)


def trending_topic_train(scraping_csv, path_save, n_features, n_topics, n_iter):
    articles = pd.read_csv(scraping_csv, sep="|")
    articles_list = articles["content"].apply(lambda x: x.lower()).tolist()
    t_topic = TrendingTopic()
    t_topic.train(articles_list, path_save, n_features, n_topics, n_iter)


def trending_topic_display(path_model, path_matrix, n_top_words):
    t_topic = TrendingTopic()
    t_topic.display(path_model, path_matrix, n_top_words)


if __name__ == "__main__":
    prepare_data()
    # article_recommender_train(
    #     "dataset/processed_scraping.csv",
    #     "models",
    # )
    # article_recommender_query(
    #     "models/recommender.pickle",
    #     "models/recommender_tfidf.csv",
    #     "Lo más relevante de la ciencia de datos en 2018",
    #     10,
    # )
    # trending_topic_train(
    #     "dataset/processed_scraping.csv",
    #     "models",
    #     500,
    #     10,
    #     10,
    # )
    # trending_topic_display(
    #     "models/trending_topic.pickle",
    #     "models/trending_topic_matrix.csv",
    #     20,
    # )
