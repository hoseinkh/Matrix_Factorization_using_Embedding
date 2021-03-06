# =========================================================
# For more info, see https://hoseinkh.github.io/projects/
# =========================================================
## Note that you need to download the file "rating.csv" ...
## ... from the following link and save it in the ...
## ... directory titled "Data".
##
## Link for the data on the kaggle:
## https://www.kaggle.com/grouplens/movielens-20m-dataset
# =========================================================
import pickle
import pandas as pd
from collections import Counter
from sklearn.utils import shuffle
## ********************************************************
## Parameters
# number of users and movies we would like to keep
n_top_users = 4000
m_top_movies = 500
## ********************************************************
df = pd.read_csv('./Data/rating.csv')
# df = df.head(2000000) # for debugging, we use smaller size of the data!
# Note:
# user ids are ordered sequentially from 1..138493
# there is no missing data, as showin below:
print(df.isna().sum())
## drop the date column (we don't need it!)
df = df.drop(columns=['timestamp'])
# since the user ids start from 1, we change it to start from 0
df['userId'] = df.userId - 1
#
#
## ********************************************************
## With a little check you can see that the movieID is ...
# ... not sequential. It is better to create a new id ...
# ... such that it is sequential.
### create a mapping for movie ids
set_unique_movie_ids = set(df.movieId.values)
#
dict_movie_to_new_id = {}
curr_new_id = 0
for curr_orig_movie_id in set_unique_movie_ids:
  dict_movie_to_new_id[curr_orig_movie_id] = curr_new_id
  curr_new_id += 1
#
# Add new moview ids to the DataFrame
df['movieId_new'] = df.apply(lambda row: dict_movie_to_new_id[row['movieId']], axis=1)
#
#
## ********************************************************
# since the size of this data set is big, and we are ...
# ... running this code on a single computer (not ...
# ... NoSQL distributed databases such as Spark), ...
# ... we are going to decrease this data set.
# We are going to keep only the most active users (i.e. ...
# ... users that watch the most movies) and the most ...
# ... watched movies.
print("original dataframe size:", len(df))
#
N_users = df['userId'].max() + 1 # number of users
M_movies = df['movieId_new'].max() + 1 # number of movies
#
## let's create a Counter (something like a dictionary) that maps ...
# ... the userId and movieId_new to the corresponding counts
user_ids_count = Counter(df['userId'])
movie_ids_count = Counter(df['movieId_new'])
#
#
top_user_ids = set([u for u, c in user_ids_count.most_common(n_top_users)])
top_movie_ids = set([m for m, c in movie_ids_count.most_common(m_top_movies)])
#
## Note that we keep only those tracks that belong to BOTH top users and top movies!
df_small = df[df['userId'].isin(top_user_ids) & df['movieId_new'].isin(top_movie_ids)].copy()
#
## Since user ids and movie ids are no longer sequential, we need to re-order them!
new_user_id_dict = dict()
curr_new_user_id = 0
for curr_old_user_id in top_user_ids:
  new_user_id_dict[curr_old_user_id] = curr_new_user_id
  curr_new_user_id += 1
#
new_movie_id_dict = dict()
curr_new_movie_id = 0
for curr_old_movie_id in top_movie_ids:
  new_movie_id_dict[curr_old_movie_id] = curr_new_movie_id
  curr_new_movie_id += 1
#
## Note that we will have
df_small.loc[:, 'userId'] = df_small.apply(lambda row: new_user_id_dict[row['userId']], axis=1)
df_small.loc[:, 'movieId_new'] = df_small.apply(lambda row: new_movie_id_dict[row['movieId_new']], axis=1)
#
df_small.rename(columns={'movieId_new': 'movieId'})
#
print("max user id:", df_small['userId'].max())
print("max movie id:", df_small['movieId_new'].max())
#
print("small dataframe size:", len(df_small))
df_small.to_csv('./Data/small_rating.csv', index=False)
#
#
#