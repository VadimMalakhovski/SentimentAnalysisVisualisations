"""
NAME
    clean_tweet

DESCRIPTION
    # Program Name: clean_tweet.py
    # Purpose: initial cleaning of raw tweets for easier read and annotation
    # Example Of: Functions of ML cleaning and formatting raw tweets
"""
import re
import pandas as pd

tweet_id, tweet_text, tweet_description = '', '', ''
in_file_name = 'tweets_data/tweets_100_02.txt'
out_file_name = 'tweets_data/tweets_removed_duplicates.txt'
tweets = ''
no_repetition_of_tweets_array = []

# hold lines already seen
lines_seen = set()
outfile = open(out_file_name, "w", encoding='utf-8')

# read the first line of tweets and stor in a var
with open(in_file_name, encoding='utf-8') as f:
    for line in f:
        tweets = line
        break


tweets = tweets.replace("[", "")
tweets = tweets.replace("]", "")
tweets_split_list = tweets.split('>,')

# pre annotation process
for t in tweets_split_list:
    tweet_text = t.split('text=')
    tweet_text = tweet_text[1]

    # remove hashtags and uri
    tweet_text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet_text).split())
    tweet_text = tweet_text.replace("https", "")
    tweet_text = tweet_text.replace("http", "")
    tweet_text = tweet_text.lower()

    # remove usernames
    tweet_text = re.sub('@[\w]+', '', tweet_text)

    #remove duplicates
    if tweet_text not in lines_seen: # not a duplicate
        outfile.write(tweet_text +'\n')
        if len(tweet_text) > 3:
            no_repetition_of_tweets_array.append(tweet_text)
        lines_seen.add(tweet_text)


df = pd.DataFrame(no_repetition_of_tweets_array)
df.columns =['Tweets']
# appending
df.to_csv('tweets1.csv', mode='a', header=False)
