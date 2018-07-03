# YelpRecommendation
Yelp Recommendation: Collaborative Filtering, XGboost, RBM, Auto-encoder

The goal of our project is to build a recommendation system to predict a rating that a user will give to a restaurant. We use dataset from Yelp Dataset Challenge (https://www.yelp.com/dataset/challenge). We applied several methods, some of which we have learned in class, to make the prediction.

The original dataset consists of 1,326,101 users, 174,567 business, and 5,261,669 reviews. Since our main focus for this project involves around suggestion of restaurants, we first reduce the dataset to have only business with "Restaurants" as one of their categories. We also filter out users who has less than 3 reviews. Then we randomly choose 5000 business. Our reduced dataset consists of 17,153 users, 4,733 business, and 90,251 reviews.

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

