#!/usr/bin/env python
# coding: utf-8

# ## Hybrid Movie Recommender System with Temporal Contextual Filtering
# 
# 
# 

# ### **Import the required Python libraries**


import pandas as pd
import seaborn as sns
import numpy as np
import json
import warnings
import base64
import io
from matplotlib.pyplot import imread
import codecs
from IPython.display import HTML
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import nltk
from nltk.corpus import stopwords
plt.style.use('fivethirtyeight')
warnings.filterwarnings('ignore')


# ### **Import the dataset**



movies_tmdb = pd.read_csv('/tmdb-movie-metadata/tmdb_5000_movies.csv')
credits = pd.read_csv('/tmdb-movie-metadata/tmdb_5000_credits.csv')
movies_movielens=pd.read_csv('/movie-lens-100k/movies.csv')
ratings_movielens=pd.read_csv('/input/movie-lens-100k/ratings.csv')


# ## **Data Exploration & Cleaning**




movies_movielens.head()



movies_movielens = movies_movielens.rename(columns={'genres': 'genre'})



ratings_movielens.head()


# **Converting JSON into strings**



# changing the genres column from json to string
movies_tmdb['genres'] = movies_tmdb['genres'].apply(json.loads)
for index,i in zip(movies_tmdb.index,movies_tmdb['genres']):
    list1 = []
    for j in range(len(i)):
        list1.append((i[j]['name'])) # the key 'name' contains the name of the genre
    movies_tmdb.loc[index,'genres'] = str(list1)

# changing the keywords column from json to string
movies_tmdb['keywords'] = movies_tmdb['keywords'].apply(json.loads)
for index,i in zip(movies_tmdb.index,movies_tmdb['keywords']):
    list1 = []
    for j in range(len(i)):
        list1.append((i[j]['name']))
    movies_tmdb.loc[index,'keywords'] = str(list1)
    
# changing the production_companies column from json to string
movies_tmdb['production_companies'] = movies_tmdb['production_companies'].apply(json.loads)
for index,i in zip(movies_tmdb.index,movies_tmdb['production_companies']):
    list1 = []
    for j in range(len(i)):
        list1.append((i[j]['name']))
    movies_tmdb.loc[index,'production_companies'] = str(list1)

# changing the cast column from json to string
credits['cast'] = credits['cast'].apply(json.loads)
for index,i in zip(credits.index,credits['cast']):
    list1 = []
    for j in range(len(i)):
        list1.append((i[j]['name']))
    credits.loc[index,'cast'] = str(list1)

# changing the crew column from json to string    
credits['crew'] = credits['crew'].apply(json.loads)
def director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
credits['crew'] = credits['crew'].apply(director)
credits.rename(columns={'crew':'director'},inplace=True)


# ### **Merging the two csv files**




movies_tmdb = movies_tmdb.merge(credits,left_on='id',right_on='movie_id',how='left')
movies_tmdb = movies_tmdb[['id','original_title','genres','cast','vote_average','director','keywords']]




duplicate_values = movies_tmdb.duplicated(subset='original_title', keep=False)

# Filter the DataFrame to keep only rows with unique values in the specified column
movies_tmdb = movies_tmdb[~duplicate_values]




duplicate_values = movies_movielens.duplicated(subset='title', keep=False)

# Filter the DataFrame to keep only rows with unique values in the specified column
movies_movielens = movies_movielens[~duplicate_values]





movies_tmdb['original_title'] = movies_tmdb['original_title'].astype(str).str.lower().str.replace(' ', '')
movies_movielens['title_key'] = movies_movielens['title'].astype(str).str.lower().str.replace(' ', '')





movies_movielens['title_key']=movies_movielens.title_key.str.replace('(\(\d\d\d\d\))','',regex=True)
movies_movielens['title_key']=movies_movielens['title_key'].apply(lambda x:x.strip())
movies_movielens['title_with_release_year']=movies_movielens['title']
movies_movielens['title']=movies_movielens.title.str.replace('(\(\d\d\d\d\))','',regex=True)
movies_movielens['title']=movies_movielens['title'].apply(lambda x:x.strip())





movies_movielens.head()





common_rows = movies_tmdb[movies_tmdb['original_title'].isin(movies_movielens['title_key'].tolist())]

# Get the number of common rows
num_common_rows = len(common_rows)
common_rows.shape




movies = pd.merge(movies_tmdb, movies_movielens, how='inner', left_on='original_title', right_on='title_key')





movies = movies.drop(['title'], axis=1)





movies = movies.drop(['genre'], axis=1)





duplicate_values = movies.duplicated(subset='original_title', keep=False)

# Filter the DataFrame to keep only rows with unique values in the specified column
movies = movies[~duplicate_values]





rating=ratings_movielens[ratings_movielens['movieId'].isin(movies['movieId'].tolist())]
rating.shape


# ## **Working with the Genres column**




movies['genres'] = movies['genres'].str.strip('[]').str.replace(' ','').str.replace("'",'')
movies['genres'] = movies['genres'].str.split(',')





plt.subplots(figsize=(12,10))
list1 = []
for i in movies['genres']:
    list1.extend(i)
ax = pd.Series(list1).value_counts()[:10].sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('hls',10))
for i, v in enumerate(pd.Series(list1).value_counts()[:10].sort_values(ascending=True).values): 
    ax.text(.8, i, v,fontsize=12,color='white',weight='bold')
plt.title('Top Genres')
plt.show()


# Drama appears to be the most popular genre followed by Comedy.




for i,j in zip(movies['genres'],movies.index):
    list2=[]
    list2=i
    list2.sort()
    movies.loc[j,'genres']=str(list2)
movies['genres'] = movies['genres'].str.strip('[]').str.replace(' ','').str.replace("'",'')
movies['genres'] = movies['genres'].str.split(',')


# Generating a list 'genreList' with all possible unique genres mentioned in the dataset.
# 
# 




genreList = []
for index, row in movies.iterrows():
    genres = row["genres"]
    
    for genre in genres:
        if genre not in genreList:
            genreList.append(genre)
genreList[:10] #now we have a list with unique genres


# **One Hot Encoding for multiple labels**




def binary(genre_list):
    binaryList = []
    
    for genre in genreList:
        if genre in genre_list:
            binaryList.append(1)
        else:
            binaryList.append(0)
    
    return binaryList





movies['genres_bin'] = movies['genres'].apply(lambda x: binary(x))
movies['genres_bin'].head()


# ## **Working with the Cast Column**
#  




movies['cast'] = movies['cast'].str.strip('[]').str.replace(' ','').str.replace("'",'').str.replace('"','')
movies['cast'] = movies['cast'].str.split(',')



plt.subplots(figsize=(12,10))
list1=[]
for i in movies['cast']:
    list1.extend(i)
ax=pd.Series(list1).value_counts()[:15].sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('muted',40))
for i, v in enumerate(pd.Series(list1).value_counts()[:15].sort_values(ascending=True).values): 
    ax.text(.8, i, v,fontsize=10,color='white',weight='bold')
plt.title('Actors with highest appearance')
plt.show()





for i,j in zip(movies['cast'],movies.index):
    list2 = []
    list2 = i[:4]
    movies.loc[j,'cast'] = str(list2)
movies['cast'] = movies['cast'].str.strip('[]').str.replace(' ','').str.replace("'",'')
movies['cast'] = movies['cast'].str.split(',')
for i,j in zip(movies['cast'],movies.index):
    list2 = []
    list2 = i
    list2.sort()
    movies.loc[j,'cast'] = str(list2)
movies['cast']=movies['cast'].str.strip('[]').str.replace(' ','').str.replace("'",'')





castList = []
for index, row in movies.iterrows():
    cast = row["cast"]
    
    for i in cast:
        if i not in castList:
            castList.append(i)





def binary(cast_list):
    binaryList = []
    
    for genre in castList:
        if genre in cast_list:
            binaryList.append(1)
        else:
            binaryList.append(0)
    
    return binaryList





movies['cast_bin'] = movies['cast'].apply(lambda x: binary(x))
movies['cast_bin'].head()


# ## **Working with Director column**




def xstr(s):
    if s is None:
        return ''
    return str(s)
movies['director'] = movies['director'].apply(xstr)




movies.head()





plt.subplots(figsize=(12,10))
ax = movies[movies['director']!=''].director.value_counts()[:10].sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('muted',40))
for i, v in enumerate(movies[movies['director']!=''].director.value_counts()[:10].sort_values(ascending=True).values): 
    ax.text(.5, i, v,fontsize=12,color='white',weight='bold')
plt.title('Directors with highest movies')
plt.show()





director_movie_counts = movies['director'].value_counts()

# Select the top 10 directors based on movie count
top_10_directors = director_movie_counts.head(10).index

# Filter the data to include only movies directed by the top 10 directors
filtered_df = movies[movies['director'].isin(top_10_directors)]

# Create a boxplot
plt.figure(figsize=(12, 6))
plt.boxplot([filtered_df[filtered_df['director'] == director]['vote_average'] for director in top_10_directors], labels=top_10_directors)
plt.title('Ratings of Movies Directed by Top 10 Directors')
plt.xlabel('Director')
plt.ylabel('Rating')
plt.xticks(rotation=45)  # Rotate x-axis labels for readability
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

plt.show()





directorList=[]
for i in movies['director']:
    if i not in directorList:
        directorList.append(i)





def binary(director_list):
    binaryList = []  
    for direct in directorList:
        if direct in director_list:
            binaryList.append(1)
        else:
            binaryList.append(0)
    return binaryList





movies['director_bin'] = movies['director'].apply(lambda x: binary(x))





MLmovie=pd.read_csv('/kaggle/input/movie-lens-100k/movies.csv')
movie3 = pd.merge(MLmovie[['movieId', 'title']], movies, on='movieId')





movies=movie3
movies.head()


# ## **Working with the Keywords column**




movies.drop(columns=['original_title'],inplace=True)
movies.rename(columns={"title_x": "original_title"}, inplace=True)










plt.subplots(figsize=(12,12))
stop_words = set(stopwords.words('english'))
stop_words.update(',',';','!','?','.','(',')','$','#','+',':','...',' ','')

words=movies['keywords'].dropna().apply(nltk.word_tokenize)
word=[]
for i in words:
    word.extend(i)
word=pd.Series(word)
word=([i for i in word.str.lower() if i not in stop_words])
wc = WordCloud(background_color="black", max_words=2000, stopwords=STOPWORDS, max_font_size= 60,width=1000,height=1000)
wc.generate(" ".join(word))
plt.imshow(wc)
plt.axis('off')
fig=plt.gcf()
fig.set_size_inches(10,10)
plt.show()


# Above is a wordcloud showing the major keywords or tags used for describing the movies.
# 



movies['keywords'] = movies['keywords'].str.strip('[]').str.replace(' ','').str.replace("'",'').str.replace('"','')
movies['keywords'] = movies['keywords'].str.split(',')
for i,j in zip(movies['keywords'],movies.index):
    list2 = []
    list2 = i
    movies.loc[j,'keywords'] = str(list2)
movies['keywords'] = movies['keywords'].str.strip('[]').str.replace(' ','').str.replace("'",'')
movies['keywords'] = movies['keywords'].str.split(',')
for i,j in zip(movies['keywords'],movies.index):
    list2 = []
    list2 = i
    list2.sort()
    movies.loc[j,'keywords'] = str(list2)
movies['keywords'] = movies['keywords'].str.strip('[]').str.replace(' ','').str.replace("'",'')
movies['keywords'] = movies['keywords'].str.split(',')





words_list = []
for index, row in movies.iterrows():
    genres = row["keywords"]
    
    for genre in genres:
        if genre not in words_list:
            words_list.append(genre)





def binary(words):
    binaryList = []
    for genre in words_list:
        if genre in words:
            binaryList.append(1)
        else:
            binaryList.append(0)
    return binaryList





movies['words_bin'] = movies['keywords'].apply(lambda x: binary(x))
movies = movies[(movies['vote_average']!=0)] #removing the movies with 0 score and without drector names 
movies = movies[movies['director']!='']


# ## Similarity between movies




from scipy import spatial

def Similarity(movieId1, movieId2):
    a = movies.iloc[movieId1]
    b = movies.iloc[movieId2]
    
    genresA = a['genres_bin']
    genresB = b['genres_bin']
    
    genreDistance = spatial.distance.cosine(genresA, genresB)
    
    scoreA = a['cast_bin']
    scoreB = b['cast_bin']
    scoreDistance = spatial.distance.cosine(scoreA, scoreB)
    
    directA = a['director_bin']
    directB = b['director_bin']
    directDistance = spatial.distance.cosine(directA, directB)
    
    wordsA = a['words_bin']
    wordsB = b['words_bin']
    wordsDistance = spatial.distance.cosine(directA, directB)
    return genreDistance + directDistance + scoreDistance + wordsDistance














new_id = list(range(0,movies.shape[0]))
movies['new_id']=new_id
movies.rename(columns={"title": "original_title"}, inplace=True)
movies=movies[['original_title','genres','vote_average','genres_bin','cast_bin','new_id','director','director_bin','words_bin']]
#movies.shape


# ## **Score Predictor**



import operator
total_neighbors = []
def KNN_predict_score(name):
#     name = input('Enter a movie title: ')
    new_movie = movies[movies['original_title'].str.contains(name)].iloc[0].to_frame().T
    def getNeighbors(baseMovie, K):
        distances = []
    
        for index, movie in movies.iterrows():
            if movie['new_id'] != baseMovie['new_id'].values[0]:
                dist = Similarity(baseMovie['new_id'].values[0], movie['new_id'])
                distances.append((movie['new_id'], dist))

        neighbors = []
    
        for x in range(K):
            neighbors.append(distances[x])
        return neighbors

    K = 10
    neighbors = getNeighbors(new_movie, K)
    total_neighbors.extend(neighbors)
    total_neighbors.sort(key=operator.itemgetter(1))


# Get all the movies for a test userId



unique_user_ids=rating['userId'].unique()



from sklearn.model_selection import train_test_split
train_userIds,test_userIds=train_test_split(unique_user_ids,test_size=0.2, random_state=42)



train_rating = pd.DataFrame(columns=rating.columns)
test_rating = pd.DataFrame(columns=rating.columns)

for userId in train_userIds:
    train_rating=pd.concat([train_rating,rating[rating['userId']==userId]],ignore_index=True)

for userId in test_userIds:
    test_rating=pd.concat([test_rating,rating[rating['userId']==userId]],ignore_index=True)


inputUserID=input("Enter User ID:")
print(inputUserID)
inputUser = test_rating[test_rating['userId']==int(inputUserID)]
inputUser.shape





inputUser.head()




from datetime import datetime,timedelta
inputUser['date'] = pd.to_datetime(inputUser['timestamp'], unit='s')
inputUser.head()





print(inputUser['date'].max())
print(inputUser['date'].min())


# ## Temporal Contextual Filtering
# 
# Filtering movies with rating>=3 given  by the input user in the most recent two years



latest_date = inputUser['date'].max()

two_years_ago = latest_date - timedelta(days=2*365)  # Approximating year as 365 days/year

inputUser = inputUser[inputUser['date'] >= two_years_ago]
inputUser.shape





print(inputUser['date'].max())
print(inputUser['date'].min())





inputUser=inputUser[inputUser['rating']>=3]
inputUser.shape




movies_movielens.head()





inputUserData = pd.merge(inputUser,movies_movielens)
print(inputUserData.shape)
inputUserData.head()





input_movies=inputUserData['title'].unique()
len(input_movies)





for movie in input_movies:
    KNN_predict_score(movie)





knncontent_output=[]
def get_recommendations():
    recommendations = total_neighbors
    print('\nMovies Recommended based on K-Nearest Neighbors Content based filtering:')
    for rec in recommendations:
        if(len(knncontent_output)>=10):
            break
        rec_movie=movies[movies['new_id']==rec[0]]
        if(not (rec_movie['original_title'].values[0] in  knncontent_output)):
            print(rec_movie['original_title'].values[0])
            knncontent_output.append(rec_movie['original_title'].values[0])





get_recommendations()





unique_user_ids=rating['userId'].unique()
len(unique_user_ids)



train_rating

test_rating





inputUserData=inputUserData.drop('genre',axis=1)




inputUserData=inputUserData.drop('title_key',axis=1)





inputUserData.head()




users=train_rating[train_rating['movieId'].isin(inputUserData['movieId'].tolist())]
users




userSubsetGroup=users.groupby(['userId'])




userSubsetGroup.head()




userSubsetGroup=sorted(userSubsetGroup,key=lambda x:len(x[1]),reverse=True)





userSubsetGroup=userSubsetGroup[0:100]





from math import sqrt

pearsonCorDic = {}

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
        pearsonCorDic[name]=Sxy/(Sxx*Syy)
    else:
        pearsonCorDic[name]=0
    




pearsonCorDic.items()




pearsonDF=pd.DataFrame.from_dict(pearsonCorDic,orient='index')
pearsonDF.columns=['similarityIndexPC']
pearsonDF['userId']=pearsonDF.index
pearsonDF.index=range(0,len(pearsonDF))
pearsonDF.head()




topUsersPC=pearsonDF.sort_values(by='similarityIndexPC',ascending=False)[0:50]
topUsersPC.head(10)





rating['userId'].info





topUsersPC['userId'] = topUsersPC['userId'].apply(lambda x: int(x[0]))





topUsersRatingPC=topUsersPC.merge(rating,left_on='userId',right_on='userId',how='inner')
topUsersRatingPC.head(300)





topUsersRatingPC['weightedRating']=topUsersRatingPC['similarityIndexPC']*topUsersRatingPC['rating']
topUsersRatingPC.head()




tempTopUsersRatingPC=topUsersRatingPC.groupby('movieId').sum()[['similarityIndexPC','weightedRating']]
tempTopUsersRatingPC.columns=['sum_similarityIndex','sum_weightedRating']
tempTopUsersRatingPC.head()





recommendation_df_pc=tempTopUsersRatingPC
recommendation_df_pc['weighted average recommendation score']=recommendation_df_pc['sum_weightedRating']/recommendation_df_pc['sum_similarityIndex']
recommendation_df_pc.head()





recommendation_df_pc[recommendation_df_pc.index.isin(inputUserData['movieId'].tolist())].shape




recommendation_df_pc



recommendation_df_pc.drop(columns=['sum_similarityIndex','sum_weightedRating'],inplace=True)
recommendation_df_pc.head()





recommendation_df_pc.sort_values(by='weighted average recommendation score',ascending=False,inplace=True)
recommendation_df_pc.shape





recommendation_df_pc.head(100)




rating_5_pc = (recommendation_df_pc['weighted average recommendation score']>=5).sum()
rating_5_pc





recommendation_df_pc1=recommendation_df_pc.head(10)
recommendation_df_pc1





movies.head()





recommendation_df_pc2=recommendation_df_pc1.merge(movies_movielens, left_on='movieId',right_on='movieId',how='inner')
recommendation_df_pc2=recommendation_df_pc2.drop(['title_key','genre'],axis=1)
recommendation_df_pc2


# # Hybrid Recommendation System




hybrid_recommender_output=[]
for i in range(5):
    hybrid_recommender_output.append(knncontent_output[i])
#     hybrid_recommender_output.append(recommendation_df_pc2.iloc[i]['title'])
    hybrid_recommender_output.append(recommendation_df_pc2.iloc[i]['title_with_release_year'])
    





print('Movies recommended by hybrid recommendation system:')
for movie in hybrid_recommender_output:
    print(movie)

