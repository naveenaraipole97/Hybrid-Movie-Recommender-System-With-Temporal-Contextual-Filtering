#!/usr/bin/env python
# coding: utf-8




import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings ("ignore")





rating = pd.read_csv('/Users/kartik/data/movielens-tmdb-merged/ratings_merged.csv')
rating.drop(columns=['Unnamed: 0'],inplace=True)
MLmovie = pd.read_csv('/Users/kartik/data/movie-lens-100k/movies.csv')
move = pd.read_csv('/Users/kartik/data/movielens-tmdb-merged/movies_merged (2).csv')
move.drop(columns=['Unnamed: 0'],inplace=True)
move2 = pd.read_csv('/Users/kartik/data/modified-100k/movies12.csv')
move2.drop(columns=['Unnamed: 0'],inplace=True)





move3 = pd.merge(MLmovie[['movieId', 'title']], move, on='movieId')
move3.drop(columns=['title_y'],inplace=True)





movie_dataset = move3[['movieId','title_x']]
movie_dataset.rename(columns={"title_x": "title"}, inplace=True)
merged_dataset = pd.merge(rating, movie_dataset, how='inner', on='movieId')
merged_dataset.head()





refined_dataset = merged_dataset.groupby(by=['userId','title'], as_index=False).agg({"rating":"mean"})
refined_dataset.head()





#list of all users
unique_users = refined_dataset['userId'].unique() 
#creating a list of all movie names in it
unique_movies = refined_dataset['title'].unique()





users_list = refined_dataset['userId'].tolist()
movie_list = refined_dataset['title'].tolist()
ratings_list = refined_dataset['rating'].tolist()





movies_dict = {unique_movies[i] : i for i in range(len(unique_movies))}


     





utility_matrix = np.asarray([[np.nan for j in range(len(unique_users))] for i in range(len(unique_movies))])
print("Shape of Utility matrix: ",utility_matrix.shape)

for i in range(len(ratings_list)):
  utility_matrix[movies_dict[movie_list[i]]][users_list[i]-1] = ratings_list[i]

utility_matrix





mask = np.isnan(utility_matrix)
masked_arr = np.ma.masked_array(utility_matrix, mask)
temp_mask = masked_arr.T
rating_means = np.mean(temp_mask, axis=0)

filled_matrix = temp_mask.filled(rating_means)
filled_matrix = filled_matrix.T
filled_matrix = filled_matrix - rating_means.data[:,np.newaxis]





filled_matrix = filled_matrix.T / np.sqrt(len(movies_dict)-1)
filled_matrix





from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from sklearn.model_selection import train_test_split

X_train, X_test, _, _ = train_test_split(filled_matrix, filled_matrix, test_size=0.2, random_state=42)

U, S, V = np.linalg.svd(X_train)

k = 50
predicted_matrix = np.dot(U[:, :k], np.dot(np.diag(S[:k]), V[:k, :]))


squared_errors = 0
absolute_errors = 0
num_predictions = 0

test_data = X_test

for i in range(test_data.shape[0]):
    for j in range(test_data.shape[1]):
        if test_data[i, j] != 0:
            squared_errors += (test_data[i, j] - predicted_matrix[i, j]) ** 2
            absolute_errors += abs(test_data[i, j] - predicted_matrix[i, j])
            num_predictions += 1

#Calculate the RMSE and MAE
rmse = np.sqrt(squared_errors / num_predictions)
mae = absolute_errors / num_predictions

print("Root Mean Squared Error (RMSE) for SVD Collaborative Filtering:", rmse)
print("Mean Absolute Error (MAE) for SVD Collaborative Filtering:", mae)





case_insensitive_movies_list = [i.lower() for i in unique_movies]





def top_cosine_similarity(data, movie_id, top_n=10):
  index = movie_id 
  movie_row = data[index, :]
  magnitude = np.sqrt(np.einsum('ij, ij -> i', data, data))
  similarity = np.dot(movie_row, data.T) / (magnitude[index] * magnitude)
  sort_indexes = np.argsort(-similarity)
  return sort_indexes[:top_n]





#k-principal components to represent movies, movie_id to find recommendations, top_n print n results        
def get_similar_movies(movie_name,top_n,k = 50):
  
  sliced = V.T[:, :k] # representative data
  movie_id = movies_dict[movie_name]
  indexes = top_cosine_similarity(sliced, movie_id, top_n)
  print(" ")
  print("Movies recommended based on SVD Collaborative Filtering for \n",movie_name, " are: ")
  print(" ")
  for i in indexes[1:]:
    print(unique_movies[i])





def get_possible_movies(movie):

    temp = ''
    possible_movies = case_insensitive_movies_list.copy()
    for i in movie :
      out = []
      temp += i
      for j in possible_movies:
        if temp in j:
          out.append(j)
      if len(out) == 0:
          return possible_movies
      out.sort()
      possible_movies = out.copy()

    return possible_movies





class invalid(Exception):
    pass

def SVD_Movie_Recommender():
    
    try:
      movie_name = "Vampires (1998)"
      #movie_name = input("Enter the Movie name: ")
      movie_name_lower = movie_name.lower()
      if movie_name_lower not in case_insensitive_movies_list :
        raise invalid
      else :
        get_similar_movies(unique_movies[case_insensitive_movies_list.index(movie_name_lower)],11)

    except invalid:

      possible_movies = get_possible_movies(movie_name_lower)

      if len(possible_movies) == len(unique_movies) :
        print("Movie name entered is does not exist in the list ")
      else :
        indices = [case_insensitive_movies_list.index(i) for i in possible_movies]
        print("Entered Movie name is not matching with any movie from the dataset . Please check the below suggestions :\n",[unique_movies[i] for i in indices])
        print("")
        SVD_Movie_Recommender()





SVD_Movie_Recommender()





rmse = np.sqrt(squared_errors / num_predictions)
mae = absolute_errors / num_predictions

print("Root Mean Squared Error (RMSE) for SVD Collaborative Filtering:", rmse)
print("Mean Absolute Error (MAE) for SVD Collaborative Filtering:", mae )

