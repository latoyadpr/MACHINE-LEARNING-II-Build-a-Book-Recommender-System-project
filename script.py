import pandas as pd
import codecademylib3
from surprise import Reader, Dataset, KNNBasic, accuracy
from surprise.model_selection import train_test_split

# Load the dataset
book_ratings = pd.read_csv('goodreads_ratings.csv')
print(book_ratings.head())

# Define similarity options
sim_options = {
    'name': 'cosine',  # or 'pearson', 'msd'
    'user_based': True  # or False for item-based
}

# Prepare data for Surprise: build a Surprise reader object
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(book_ratings[['user_id', 'book_id', 'rating']], reader)

# Create an 80:20 train-test split and set the random state to 7
trainset, testset = train_test_split(data, test_size=0.2, random_state=7)

# Create an instance of KNNBasic with custom parameters
algo = KNNBasic(k=20, sim_options=sim_options)

# Train the algorithm on the training set
algo.fit(trainset)

# Test the algorithm on the test set
predictions = algo.test(testset)

# Calculate RMSE
rmse = accuracy.rmse(predictions)

# Print the distribution of ratings
print(book_ratings['rating'].value_counts())
