import tweepy
import pandas as pd
import requests
import re 
from flair.models import TextClassifier
from flair.data import Sentence
import time
from ascii_graph import Pyasciigraph
import os

sia = TextClassifier.load('en-sentiment')

# TWIITER API keys
api_key = "q99GLCecgjGMC6nxpA6zBfJug"
api_secrets = "rzvcxcd3M7X9vi7idjOvY7bGGecPP8ii6EX8nlAusEhMHcJaac"
access_token = "1545188617483599873-GZnDWrpgSEojmGyU5vwN1xYoVtQs6p"
access_secret = "uTyrFKZjQrVSc3PBjTJ6lfCzTPioD3Kb8MTy857V8hObm"
b_token = 'AAAAAAAAAAAAAAAAAAAAAE5QegEAAAAA30Bq7iBhdVoHedvlasOQePdwynQ%3D9unApLm8hZKXmd0lLyiFJzlBH9OfX5PgFVC2zgGj4SQ5O9MK59'



def get_client():
    try:
        client = tweepy.Client(bearer_token=b_token,return_type = requests.Response)
        print('Successful Authentication')
        return client
    except Exception as e:
        print('Failed authentication')
        print(e)
        return False

def sentiment_analysis(tweet):
    sentence = Sentence(tweet)
    sia.predict(sentence)
    analysis = str(sentence.labels[0])
    analysis_split = re.split("NEGATIVE | POSITIVE",analysis)
    
    score = analysis_split[-1]
    score = float(score.strip(" ").strip("()"))
        
    if "NEGATIVE" in str(analysis):
        score = score * -1

    return score

def get_tweets(client,q,n):

    tweets = client.search_recent_tweets(q,max_results=n,tweet_fields=['public_metrics'])

    tweets_dict = tweets.json() 

    # Extract "data" value from dictionary
    tweets_data = tweets_dict['data'] 

    # Transform to pandas Dataframe
    df = pd.json_normalize(tweets_data)

    return df

def display(df):
    
    os.system('cls' if os.name == 'nt' else 'clear')
    print(df['sentiment'].mean())
    weighted_sentiment = (df["sentiment"].mul(df['public_metrics.retweet_count']).sum()) / df['public_metrics.retweet_count'].sum()
    print(weighted_sentiment)
    data = [('Sentiment', df['sentiment'].mean()*100),('Weighted Sentiment', weighted_sentiment*100)]

    graph = Pyasciigraph(force_max_value=100)  
    for line in  graph.graph('Market Sentiment', data):
        print(line)

def cls():
    os.system('cls' if os.name=='nt' else 'clear')

def main():
    client = get_client()  
    
    while True:
        df = get_tweets(client,"Ethereum",100)
        df["sentiment"] = df["text"].apply(sentiment_analysis)
               
        display(df)
        
        
    # save twets with sentiment to  csv
    print(df.head())
    df.to_csv('out.csv')


if __name__ == "__main__":
    print("---Running Setup---")
    main()