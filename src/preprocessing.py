import pandas as pd

def load_data():
    data = pd.read_csv("data/imdb.csv")
    return data

def clean_data(data):
    return data.dropna()