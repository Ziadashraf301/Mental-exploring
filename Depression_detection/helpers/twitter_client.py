import re
import pandas as pd
from textblob import TextBlob
import tweepy as tw
from tweepy import OAuthHandler

class TwitterClient:
    """
    Generic Twitter Class for sentiment analysis.
    """

    def __init__(self, consumer_key, consumer_secret, access_token, access_token_secret):
        """
        Class constructor: initializes Twitter API authentication.
        """
        try:
            # create OAuthHandler object
            self.auth = OAuthHandler(consumer_key, consumer_secret)
            # set access token and secret
            self.auth.set_access_token(access_token, access_token_secret)
            # create tweepy API object to fetch tweets
            self.api = tw.API(self.auth)
        except Exception as e:
            print("Error: Authentication Failed", e)

    def clean_tweet(self, tweet):
        """
        Cleans tweet text by removing links, special characters, mentions, etc.
        """
        tweet = ' '.join(re.sub(
            "(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet
        ).split())
        return tweet

    def get_tweet_sentiment(self, tweet):
        """
        Classify sentiment of tweet using TextBlob.
        """
        analysis = TextBlob(tweet)
        if analysis.sentiment.polarity > 0.3:
            return 'positive'
        elif analysis.sentiment.polarity > -0.3:
            return 'neutral'
        else:
            return 'negative'

    def get_tweets(self, query, count=10):
        """
        Fetch tweets and parse them.
        Returns a list of dicts with 'text' and 'sentiment'.
        """
        tweets = []

        try:
            fetched_tweets = self.api.search_tweets(q=query, count=count)
            for tweet in fetched_tweets:
                parsed_tweet = {}
                cleaned_tweet = self.clean_tweet(tweet.text)
                parsed_tweet['text'] = cleaned_tweet
                parsed_tweet['sentiment'] = self.get_tweet_sentiment(cleaned_tweet)

                # append only unique tweets with retweets
                if tweet.retweet_count > 0:
                    if parsed_tweet not in tweets:
                        tweets.append(parsed_tweet)
                else:
                    tweets.append(parsed_tweet)

            return tweets

        except tw.errors.TweepyException as e:
            print("Error fetching tweets:", e)
            return []

    def tweet_labels_stat(self, tweets):
        """
        Print statistics of positive, negative, and neutral tweets.
        Returns a DataFrame with all tweets.
        """
        self.ptweets = [tweet for tweet in tweets if tweet['sentiment'] == 'positive']
        self.ntweets = [tweet for tweet in tweets if tweet['sentiment'] == 'negative']
        self.neutral_tweets = [tweet for tweet in tweets if tweet['sentiment'] == 'neutral']

        print("Positive tweets percentage: {}%".format(100 * len(self.ptweets) / len(tweets)))
        print("Negative tweets percentage: {}%".format(100 * len(self.ntweets) / len(tweets)))
        print("Neutral tweets percentage: {}%".format(100 * len(self.neutral_tweets) / len(tweets)))

        self.data = pd.DataFrame(tweets)
        return self.data
