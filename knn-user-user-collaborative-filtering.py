#!/usr/bin/env python
# coding: utf-8




import numpy as np
import pandas as pd
from math import sqrt 
import warnings
warnings.filterwarnings('ignore')





movie_movielens = pd.read_csv('/movie-lens-100k/movies.csv')
movielens_temp=pd.read_csv('/movie-lens-100k/movies.csv')
rating = pd.read_csv('/movie-lens-100k/ratings.csv')
movie_tmdb=pd.read_csv('/tmdb-movie-metadata/tmdb_5000_movies.csv')





movie_movielens.head()





movie_movielens.shape





rating.head()





rating.shape





movie_tmdb.shape





movie_tmdb.head()





movielens_temp['year']=movielens_temp.title.str.extract('(\d\d\d\d)',expand=False)
movie_movielens['year']=movie_movielens.title.str.extract('(\d\d\d\d)',expand=False)
movie_movielens.head()





movielens_temp['title']=movielens_temp.title.str.replace('(\(\d\d\d\d\))','',regex=True)
movielens_temp['title']=movielens_temp['title'].apply(lambda x:x.strip())
movie_movielens['title']=movie_movielens.title.str.replace('(\(\d\d\d\d\))','',regex=True)
movie_movielens['title']=movie_movielens['title'].apply(lambda x:x.strip())
movie_movielens.head()





movie_tmdb['original_title'] = movie_tmdb['original_title'].astype(str).str.lower().str.replace(' ', '')
movie_movielens['title'] = movie_movielens['title'].astype(str).str.lower().str.replace(' ', '')





len(movie_movielens['title'].unique())





len(movie_tmdb['original_title'].unique())





duplicate_values = movie_tmdb.duplicated(subset='original_title', keep=False)

# Filter the DataFrame to keep only rows with unique values in the specified column
movie_tmdb = movie_tmdb[~duplicate_values]





movie_tmdb.shape





duplicate_values = movie_movielens.duplicated(subset='title', keep=False)

# Filter the DataFrame to keep only rows with unique values in the specified column
movie_movielens = movie_movielens[~duplicate_values]





movie_movielens.shape





movie=movie_movielens[movie_movielens['title'].isin(movie_tmdb['original_title'].tolist())]
movie.shape





movie=movielens_temp[movielens_temp['movieId'].isin(movie['movieId'].tolist())]
movie.shape





movie.head()





movie.shape





len(movie.index)





movie.drop(columns=['genres'],inplace=True)
movie.head()





rating=rating[rating['movieId'].isin(movie['movieId'].tolist())]
rating.shape





rating.head()





rating.drop(columns=['timestamp'],inplace=True)





rating.head()





unique_user_ids=rating['userId'].unique()
len(unique_user_ids)





from sklearn.model_selection import train_test_split
train_userIds,test_userIds=train_test_split(unique_user_ids,test_size=0.2, random_state=42)
len(train_userIds)





len(test_userIds)





rating.head()





train_rating = pd.DataFrame(columns=rating.columns)
test_rating = pd.DataFrame(columns=rating.columns)

for userId in train_userIds:
    train_rating=pd.concat([train_rating,rating[rating['userId']==userId]],ignore_index=True)

for userId in test_userIds:
    test_rating=pd.concat([test_rating,rating[rating['userId']==userId]],ignore_index=True)





train_rating





test_rating





#Create user profile from user ID (get it from data set)
import random
inputUserID = random.choice(test_userIds)
print(inputUserID)
inputUser = test_rating[test_rating['userId']==inputUserID]
inputUser.shape





inputUser.head()





movie.head()





inputUserData = pd.merge(inputUser,movie)
inputUserData.head()





train_rating.head()





users=train_rating[train_rating['movieId'].isin(inputUserData['movieId'].tolist())]
users





users.shape





#this step is not required as train and test data are different
users=users.drop(users[users['userId']==inputUserID].index)
users.shape 





userSubsetGroup=users.groupby(['userId'])





userSubsetGroup.head()





userSubsetGroup=sorted(userSubsetGroup,key=lambda x:len(x[1]),reverse=True)





userSubsetGroup[0:1]





inputUserData.head()





from scipy.spatial.distance import cosine


pearsonCorDic = {}
cosineSimilarityDic = {}

for name, group in userSubsetGroup:
    group=group.sort_values(by='movieId') #here they are already sorted
    inputUserData=inputUserData.sort_values(by='movieId') #here they are already in sorted order
    
    n=len(group)
    
    #Get the ratings for the movies that are common 
    temp=inputUserData[inputUserData['movieId'].isin(group['movieId'].tolist())]
    
    tempRatingList=temp['rating'].tolist()
    tempGroupList=group['rating'].tolist()
        
    x_mean=sum(tempRatingList)/float(n)
    y_mean=sum(tempGroupList)/float(n)
    
    Sxx= sqrt(sum(pow((i-x_mean),2) for i in tempRatingList))
    Syy= sqrt(sum(pow((i-y_mean),2) for i in tempGroupList))
    
    Sxy= sum((tempRatingList[i]-x_mean)*(tempGroupList[i]-y_mean) for i in range(0,n))
    
    #if denominator is not 0, then divide else 0 correlation
    
    if Sxx!=0 and Syy!=0:
        pearsonCorDic[name[0]]=Sxy/(Sxx*Syy)
    else:
        pearsonCorDic[name[0]]=0
        
    #Calculating cosine similarity
    vector_a = np.array(tempRatingList)
    vector_b = np.array(tempGroupList)
    
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    
    vector_a_normalized = vector_a / norm_a
    vector_b_normalized = vector_b / norm_b
    
    cosine_similarity = np.dot(vector_a_normalized,vector_b_normalized)
    
    cosineSimilarityDic[name[0]]=cosine_similarity
    





pearsonCorDic.items()





cosineSimilarityDic.items()





pearsonDF=pd.DataFrame.from_dict(pearsonCorDic,orient='index')
pearsonDF.columns=['similarityIndexPC']
pearsonDF['userId']=pearsonDF.index
pearsonDF.index=range(0,len(pearsonDF))
pearsonDF.head()





cosineDF=pd.DataFrame.from_dict(cosineSimilarityDic,orient='index')
cosineDF.columns=['similarityIndexCS']
cosineDF['userId']=cosineDF.index
cosineDF.index=range(0,len(cosineDF))
cosineDF.head()





topUsersPC=pearsonDF.sort_values(by='similarityIndexPC',ascending=False)[0:50]
topUsersPC.head(10)





topUsersCS=cosineDF.sort_values(by='similarityIndexCS',ascending=False)[0:50]
topUsersCS.head(10)





rating.head()





topUsersRatingPC=topUsersPC.merge(rating,left_on='userId',right_on='userId',how='inner')
topUsersRatingPC.head(300)





topUsersRatingCS=topUsersCS.merge(rating,left_on='userId',right_on='userId',how='inner')
topUsersRatingCS.head()





topUsersRatingPC['weightedRating']=topUsersRatingPC['similarityIndexPC']*topUsersRatingPC['rating']
topUsersRatingPC.head()





topUsersRatingCS['weightedRating']=topUsersRatingCS['similarityIndexCS']*topUsersRatingCS['rating']
topUsersRatingCS.head()





tempTopUsersRatingPC=topUsersRatingPC.groupby('movieId').sum()[['similarityIndexPC','weightedRating']]
tempTopUsersRatingPC.columns=['sum_similarityIndex','sum_weightedRating']
tempTopUsersRatingPC.head()





tempTopUsersRatingCS=topUsersRatingCS.groupby('movieId').sum()[['similarityIndexCS','weightedRating']]
tempTopUsersRatingCS.columns=['sum_similarityIndex','sum_weightedRating']
tempTopUsersRatingCS.head()





recommendation_df_pc=tempTopUsersRatingPC
recommendation_df_pc['weighted average recommendation score']=recommendation_df_pc['sum_weightedRating']/recommendation_df_pc['sum_similarityIndex']
recommendation_df_pc.head()





recommendation_df_pc.shape





inputUserData.shape





inputUserData.head()





recommendation_df_pc[recommendation_df_pc.index.isin(inputUserData['movieId'].tolist())].shape





df_accuracy_pc=recommendation_df_pc[recommendation_df_pc.index.isin(inputUserData['movieId'].tolist())]
df_accuracy_pc.head()





recommendation_df_cs=tempTopUsersRatingCS
recommendation_df_cs['weighted average recommendation score']=recommendation_df_cs['sum_weightedRating']/recommendation_df_cs['sum_similarityIndex']
recommendation_df_cs.head()





recommendation_df_cs.shape





recommendation_df_pc.drop(columns=['sum_similarityIndex','sum_weightedRating'],inplace=True)
recommendation_df_pc.head()





recommendation_df_cs.drop(columns=['sum_similarityIndex','sum_weightedRating'],inplace=True)
recommendation_df_cs.head()





recommendation_df_cs.shape





df_accuracy_cs=recommendation_df_cs[recommendation_df_cs.index.isin(inputUserData['movieId'].tolist())]
df_accuracy_cs.head()





recommendation_df_pc.sort_values(by='weighted average recommendation score',ascending=False,inplace=True)
recommendation_df_pc.shape





recommendation_df_pc.tail()





inputUserData.head()





recommendation_df_pc.head(100)





rating_5_pc = (recommendation_df_pc['weighted average recommendation score']>=5).sum()
rating_5_pc





recommendation_df_cs.sort_values(by='weighted average recommendation score',ascending=False,inplace=True)
recommendation_df_cs.head(10)





rating_5_cs = (recommendation_df_cs['weighted average recommendation score']>=5).sum()
rating_5_cs





recommendation_df_pc1=recommendation_df_pc.head(10)
recommendation_df_pc1





recommendation_df_cs1=recommendation_df_cs.head(10)
recommendation_df_cs1





recommendation_df_pc2=recommendation_df_pc1.merge(movie, left_on='movieId',right_on='movieId',how='inner')
print("Top 10 recommended movies with knn using pearson coefficient as distance metric:\n")
recommendation_df_pc2





inputUserData.head()





df_accuracy_pc.head()





df_accuracy_pc=df_accuracy_pc.sort_index()





df_accuracy_pc.head()





inputUserData_accuracy_pc=inputUserData[inputUserData['movieId'].isin(df_accuracy_pc.index.tolist())]





inputUserData_accuracy_pc=inputUserData_accuracy_pc.sort_values(by='movieId')





inputUserData_accuracy_pc.head()





inputUserData_accuracy_pc.index





inputUserData_accuracy_pc.index = range(len(inputUserData_accuracy_pc))





inputUserData_accuracy_pc.head()





df_accuracy_pc.index





df_accuracy_pc['movieId']=df_accuracy_pc.index
df_accuracy_pc.head()





df_accuracy_pc.index=range(len(df_accuracy_pc))
df_accuracy_pc.head()





df_accuracy_pc.shape





sum_sd=0
sum_ad=0
for i in range(df_accuracy_pc.shape[0]):
    if(inputUserData_accuracy_pc.iloc[i]['movieId']==df_accuracy_pc.iloc[i]['movieId']):
        sum_sd+=(inputUserData_accuracy_pc.iloc[i]['rating']-df_accuracy_pc.iloc[i]['weighted average recommendation score'])**2
        sum_ad+=abs(inputUserData_accuracy_pc.iloc[i]['rating']-df_accuracy_pc.iloc[i]['weighted average recommendation score'])

rmse=sqrt(sum_sd/df_accuracy_pc.shape[0])
print("rmse of movie recommendations with knn using pearson coefficient:", rmse)

mae=sum_ad/df_accuracy_pc.shape[0]
print("mae of movie recommendations with knn using pearson coefficient:", mae)





recommendation_df_cs2=recommendation_df_cs1.merge(movie, left_on='movieId',right_on='movieId',how='inner')
print("Top 10 recommended movies with knn using cosine similarity as distance metric:\n")
recommendation_df_cs2





df_accuracy_cs.head()





df_accuracy_cs.index





inputUserData_accuracy_cs=inputUserData[inputUserData['movieId'].isin(df_accuracy_cs.index.tolist())]
inputUserData_accuracy_cs.head()





df_accuracy_cs['movieId']=df_accuracy_cs.index
df_accuracy_cs.head()





df_accuracy_cs.index=range(len(df_accuracy_cs))
df_accuracy_cs.head()





inputUserData_accuracy_cs.head()





inputUserData_accuracy_cs=inputUserData_accuracy_cs.sort_values(by='movieId')
inputUserData_accuracy_cs.head()





inputUserData_accuracy_cs.index=range(len(inputUserData_accuracy_cs))
inputUserData_accuracy_cs.head()





df_accuracy_cs=df_accuracy_cs.sort_values(by='movieId')
df_accuracy_cs.head()





sum_sd=0
sum_ad=0
for i in range(df_accuracy_cs.shape[0]):
    if(inputUserData_accuracy_cs.iloc[i]['movieId']==df_accuracy_cs.iloc[i]['movieId']):
        sum_sd+=(inputUserData_accuracy_cs.iloc[i]['rating']-df_accuracy_cs.iloc[i]['weighted average recommendation score'])**2
        sum_ad+=abs(inputUserData_accuracy_cs.iloc[i]['rating']-df_accuracy_cs.iloc[i]['weighted average recommendation score'])
        
rmse=sqrt(sum_sd/df_accuracy_cs.shape[0])
print("rmse of movie recommendations with knn using cosine similarity between users:", rmse)

mae=sum_ad/df_accuracy_cs.shape[0]
print("mae of movie recommendations with knn using cosine similarity between users:", mae)







