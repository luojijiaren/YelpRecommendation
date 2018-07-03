# YelpRecommendation
Yelp Recommendation: Collaborative Filtering, XGboost, RBM, Auto-encoder

I build a recommendation system to predict a rating that a user will give to a restaurant. I use dataset from Yelp Dataset Challenge (https://www.yelp.com/dataset/challenge). 

The original dataset consists of 1,326,101 users, 174,567 business, and 5,261,669 reviews. 

There are four parts:
1. Data exploration
Code 'dataExploration'

2.Applying Collaborative Filtering Algorithm, Latent Factor Model
Code 'CF_RecommendationsALS.scala', 'CollaborativeFilter.py', 'lfm-tf.py'

3.Applying AutoEncoder, Restricted Boltzmann Machine (RBM)
Code 'AE_final.py', 'RBM_final.py'.

4. Using other available attributes to make a prediction, Logistic Regression, XGBoost
Code 'MachineLearningmain.py'

5. Combined all the models to a voting classfier. 
Code 'combine.py' and 'main.py

