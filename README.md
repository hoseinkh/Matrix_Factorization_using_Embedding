# Matrix Factorization using Embedding Layers

Here we implement Matrix Factorization using embedding layers.

<br />

## Task:

The goal is to derive latent representation of the user and item feature vectors. The (predicted) ratings that a user gives to an item is the inner product of user's latent vector and the item's latent vector.

The idea here is to use the technique that we often used when we want to convert categorical variables into numerical vectors: **Embedding**. *Keras* has embedding layers that automatically learns the latent representation of such cases. 

We use the 20 million MovieLens data set available on [Kaggle](https://www.kaggle.com/grouplens/movielens-20m-dataset). Though, for practical implementation on a pc we shrink this dataset.

---

Matrix factorization plays a major role in the recommender systems. It:

- decreases the computations
- improves the performance as it increases the robustness of the system w.r.t. the noise.

The process of the matrix factorization using embedding layers is shown in the following figure (the original figure is from this [reference](https://www.kaggle.com/code/colinmorris/matrix-factorization/notebook))

<p float="left">
  <img src="/figs/MF_embedding_form.png" width="450" />
</p>



---

### Codes & Results

The code consist of two parts. One is for the data preprocessing, and one implements and matrix factorization using embedding layer, and gets the results.

<p float="left">
  <img src="/figs/MF_embedding_loss_for_train_and_test.png" width="450" />
</p>








------

### References

1. [Recommender Systems Handbook; Ricci, Rokach, Shapira](https://www.cse.iitk.ac.in/users/nsrivast/HCC/Recommender_systems_handbook.pdf)
2. [Statistical Methods for Recommender Systems; Agarwal, Chen](https://www.cambridge.org/core/books/statistical-methods-for-recommender-systems/0051A5BA0721C2C6385B2891D219ECD4)

