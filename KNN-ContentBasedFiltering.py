



import pandas as pd
import seaborn as sns
import numpy as np
import json
import warnings
import base64
import io
from scipy import spatial
from matplotlib.pyplot import imread
import codecs
from IPython.display import HTML
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
warnings.filterwarnings('ignore')





movies_tmdb = pd.read_csv('/Users/kartik/data/tmdb/tmdb_5000_movies.csv')
credits = pd.read_csv('/Users/kartik/data/tmdb/tmdb_5000_credits.csv')
movies_movielens=pd.read_csv('/Users/kartik/data/movie-lens-100k/movies.csv')
ratings_movielens=pd.read_csv('/Users/kartik/data/movie-lens-100k/ratings.csv')





movies_movielens.columns = ['movieId','title','genre']





ratings_movielens.columns=['userId','movieId','rating','timestamp']





ratings_movielens.head()





unique_user_ids=ratings_movielens['userId'].unique()
len(unique_user_ids)





#Changing the column Data from json to string format
movies_tmdb['genres'] = movies_tmdb['genres'].apply(json.loads)
for index,i in zip(movies_tmdb.index,movies_tmdb['genres']):
    list1 = []
    for j in range(len(i)):
        list1.append((i[j]['name'])) # the key 'name' contains the name of the genre
    movies_tmdb.loc[index,'genres'] = str(list1)

movies_tmdb['keywords'] = movies_tmdb['keywords'].apply(json.loads)
for index,i in zip(movies_tmdb.index,movies_tmdb['keywords']):
    list1 = []
    for j in range(len(i)):
        list1.append((i[j]['name']))
    movies_tmdb.loc[index,'keywords'] = str(list1)
    
movies_tmdb['production_companies'] = movies_tmdb['production_companies'].apply(json.loads)
for index,i in zip(movies_tmdb.index,movies_tmdb['production_companies']):
    list1 = []
    for j in range(len(i)):
        list1.append((i[j]['name']))
    movies_tmdb.loc[index,'production_companies'] = str(list1)


credits['cast'] = credits['cast'].apply(json.loads)
for index,i in zip(credits.index,credits['cast']):
    list1 = []
    for j in range(len(i)):
        list1.append((i[j]['name']))
    credits.loc[index,'cast'] = str(list1)
 
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
movies_movielens['title'] = movies_movielens['title'].astype(str).str.lower().str.replace(' ', '')





movies_movielens['title']=movies_movielens.title.str.replace('(\(\d\d\d\d\))','',regex=True)
movies_movielens['title']=movies_movielens['title'].apply(lambda x:x.strip())





common_rows = movies_tmdb[movies_tmdb['original_title'].isin(movies_movielens['title'].tolist())]

# Get the number of common rows
num_common_rows = len(common_rows)
common_rows.shape





movies = pd.merge(movies_tmdb, movies_movielens, how='inner', left_on='original_title', right_on='title')
movies = movies.drop(['title'], axis=1)
movies = movies.drop(['genre'], axis=1)





duplicate_values = movies.duplicated(subset='original_title', keep=False)

# Filter the DataFrame to keep only rows with unique values in the specified column
movies = movies[~duplicate_values]





ratings_movielens=pd.merge(ratings_movielens, movies_movielens, how='inner', left_on='movieId', right_on='movieId')





ratings_movielens=ratings_movielens[ratings_movielens['title'].isin(movies_tmdb['original_title'].tolist())]





ratings_movielens=ratings_movielens.drop(['timestamp','genre'],axis=1)





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





for i,j in zip(movies['genres'],movies.index):
    list2=[]
    list2=i
    list2.sort()
    movies.loc[j,'genres']=str(list2)
movies['genres'] = movies['genres'].str.strip('[]').str.replace(' ','').str.replace("'",'')
movies['genres'] = movies['genres'].str.split(',')





genreList = []
for index, row in movies.iterrows():
    genres = row["genres"]
    
    for genre in genres:
        if genre not in genreList:
            genreList.append(genre)
genreList[:10] #now we have a list with unique genres





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
top_10_directors = director_movie_counts.head(10).index

# Filter the data to include only movies directed by the top 10 directors
filtered_df = movies[movies['director'].isin(top_10_directors)]

#Printin BoxPlot
plt.figure(figsize=(12, 6))
plt.boxplot([filtered_df[filtered_df['director'] == director]['vote_average'] for director in top_10_directors], labels=top_10_directors)
plt.title('Ratings of Movies Directed by Top 10 Directors')
plt.xlabel('Director')
plt.ylabel('Rating')
plt.xticks(rotation=45)  
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
move3 = pd.merge(MLmovie[['movieId', 'title']], movies, on='movieId')
movies=move3
movies.head()





movies.drop(columns=['original_title'],inplace=True)
movies.rename(columns={"title_x": "original_title"}, inplace=True)





from wordcloud import WordCloud, STOPWORDS
import nltk
from nltk.corpus import stopwords





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
#This will print a wordcloud showing the major keywords or tags used for describing the movies.





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





#We will we using Cosine Similarity for finding the similarity between 2 movies.
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

def KNN_predict_score():
    name = input('Enter a movie title: ')
    new_movie = movies[movies['original_title'].str.contains(name)].iloc[0].to_frame().T
    print('Selected Movie: ',new_movie.original_title.values[0])
    def getNeighbors(baseMovie, K):
        distances = []
    
        for index, movie in movies.iterrows():
            if movie['new_id'] != baseMovie['new_id'].values[0]:
                dist = Similarity(baseMovie['new_id'].values[0], movie['new_id'])
                distances.append((movie['new_id'], dist))
    
        distances.sort(key=operator.itemgetter(1))
        neighbors = []
    
        for x in range(K):
            neighbors.append(distances[x])
        return neighbors

    K = 10
    avgRating = 0
    neighbors = getNeighbors(new_movie, K)
    sigm = 0
    mae=0
    print('\nMovies Recommended based on K-Nearest Neighbors Content based filtering:')
    for neighbor in neighbors:
        avgRating = avgRating+movies.iloc[neighbor[0]][2] 
        mae+= abs(float(new_movie['vote_average'])-float(movies.iloc[neighbor[0]][2]))
        sigm += pow(float(new_movie['vote_average'])-float(movies.iloc[neighbor[0]][2]),2)
        print( movies.iloc[neighbor[0]][0])
    
    print('\n')
    avgRating = avgRating/K
    sigm = (sigm/K)**(0.5)
    mae = mae/K
    
    print('Root mean squared error for KNN Content Based Filtering =',sigm)
    print('Mean Absolute Error for KNN Content Based Filtering =',mae)
   





KNN_predict_score()

