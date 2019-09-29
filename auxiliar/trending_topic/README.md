# Trending Topic

The purpose of this class is to train and display the main topics over all the articles

## Training

In order to identify the trending topics in our blogs, we will use a topic modeling algorithm. For this implementation we start from finding how important is a word to a document in a collection of documents by calculating the term frequency TF-IDF. First we started with some data preprocessing techniques.

We trained a model where we only use the top 500 words and we want to see only the top 10 topics.

Since it is an unsupervised model, we need identify which topics were found. Due to the not optimized parameters these topics might not
be easy to define.

## Which topics are we missing?

For this part we will use a pretrained word embedding in order to return the most similar topics to our topics.
Since some of them are available online, we will not implement it as our part of our code.
In this case we used sense2vec which demo is available here: https://explosion.ai/demos/sense2vec
In this wordembedding we can query any doc and it will calculate the most similar topics.
When we query, for instance, 'data analytics', the most similar topics are:


| Topic                | similarity |
| -------------------- | ---------- |
| data science         | 86%        |
| big data             | 83%        |
| data analysis        | 83%        |
| software development | 83%        |
| business intelligence| 82%        |
| database management  | 81%        |
| project management   | 81%        |
| machine learning     | 80%        |
| web development      | 80%        |
| database development | 80%        |
