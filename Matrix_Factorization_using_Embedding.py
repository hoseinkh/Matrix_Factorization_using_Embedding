# =========================================================
# For more info, see https://hoseinkh.github.io/projects/
# =========================================================
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dot, Add, Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD, Adam
## ********************************************************
## Loading the data
## Note that we need to first run the file "data_preprocessing.py" to ...
# ... get the smaller version of the data set.
# If you want to perform the computations on the original data, ...
# ... then replace "small_rating.csv" with the "rating.csv" in the following line.
df = pd.read_csv('./Data/small_rating.csv')
## ********************************************************
## Hyperparameters:
K = 10 # latent dimensionality
epochs = 25
reg = 0 # regularization penalty
train_dataset_ratio = 0.8
## ********************************************************
#
N_max_user_id_in_train = df["userId"].max() + 1 # number of users
M_max_movie_id_in_tain_and_test = df["movieId"].max() + 1 # number of movies
#
## ****************************
## split into train and test
df = shuffle(df)
cutoff = int(train_dataset_ratio*len(df))
df_train = df.iloc[:cutoff]
df_test = df.iloc[cutoff:]
## ****************************
## calculating the global bias term
mu = df_train.rating.mean()
#
## ********************************************************
## Defining keras model
u = Input(shape=(1,)) # this means that the input is of size: batch_size * 1
m = Input(shape=(1,)) # this means that the input is of size: batch_size * 1
#
# defining the user and movie embedding layers
u_embedding = Embedding(N_max_user_id_in_train, K, embeddings_regularizer=l2(reg))(u) # output shape: (N_max_user_id_in_train, 1, K)
m_embedding = Embedding(M_max_movie_id_in_tain_and_test, K, embeddings_regularizer=l2(reg))(m) # output shape: (N, 1, K)
#
# defining the bias layers
u_bias = Embedding(N_max_user_id_in_train, 1, embeddings_regularizer=l2(reg))(u) # output shape: (N_max_user_id_in_train, 1, 1)
m_bias = Embedding(M_max_movie_id_in_tain_and_test, 1, embeddings_regularizer=l2(reg))(m) # output shape: (N_max_user_id_in_train, 1, 1)
#
#
## ****************************
## Define the model
## here we start calculatibg the rating based on the embedded vectors
# in the following line, the axis=2 indicates to calculate the dot product ...
# ... on the embedded dimension (i.e. dimension with k features)
x = Dot(axes=2)([u_embedding, m_embedding]) # output shape: (N_max_user_id_in_train, 1, 1)
#
## add the user_bias term and movie_bias_term to the x
x = Add()([x, u_bias, m_bias]) # output shape: (N_max_user_id_in_train, 1, 1)
## remove the extra dimension
x = Flatten()(x) # output shape: (N_max_user_id_in_train, 1)
#
## ****************************
## Build the model
model = Model(inputs=[u, m], outputs=x)
model.compile(
  loss='mse',
  # optimizer='adam',
  # optimizer=Adam(lr=0.01),
  optimizer=SGD(lr=0.08, momentum=0.9),
  metrics=['mse'],
)
## ********************************************************
## Training the model
r = model.fit(
  x=[df_train["userId"].values, df_train["movieId"].values],
  y=df_train["rating"].values - mu,
  epochs=epochs,
  batch_size=128,
  validation_data=(
    [df_test["userId"].values, df_test["movieId"].values],
    df_test["rating"].values - mu
  )
)
## ********************************************************
# plot Loss for train and test
plt.plot(r.history['loss'], label="train loss")
plt.plot(r.history['val_loss'], label="test loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.savefig("./figs/loss_for_train_and_test.png")
plt.show()







