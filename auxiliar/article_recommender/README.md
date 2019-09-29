# Article Recommender

The purpose of this class is to train and query an article recommender engine.

## Training

The training process consist in reading all the available blog documents.
Since the documents are noisy, some preprocessing is needed, this was done by using regular.
expressions

## Query

For the query, in order to avoid a possible bias due to long documents, a double normalization was used 
with a value of 0.5 for the hyperparameter.

The query can be adapted to receive an article object where it can query by using the entire document
and not only the title or keywords.

