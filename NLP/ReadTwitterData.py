import os
import tweepy as tw
import pandas as pd
import datetime


class ReadTwitterData:

    def __init__(self, key):
        self.consumer_key = key[0]
        self.consumer_secret = key[1]
        self.access_token = key[2]
        self.access_token_secret = key[3]
        self.auth = tw.OAuthHandler(self.consumer_key, self.consumer_secret)
        self.auth.set_access_token(self.access_token, self.access_token_secret)
        self.api = tw.API(self.auth, wait_on_rate_limit=True)

    # Get user tweets and respective tweet ids - last 50
    def get_user_timeline(self, count=50):
        user_tweets = []
        tweets = self.api.user_timeline(tweet_mode="extended", count=count)
        for tweet in tweets:
            try:
                user_tweets.append([tweet.retweeted_status.full_text, tweet.id])
            except AttributeError:  # Not a Retweet
                user_tweets.append([tweet.full_text, tweet.id])
        user_tweets = pd.DataFrame(user_tweets)
        return user_tweets

    # Get user mentions, user id, tweet ids - last 50
    def get_user_mentions(self, count=50):
        user_mentions = []
        tweets_mentions = self.api.mentions_timeline(tweet_mode="extended", count=count)
        for tweet in tweets_mentions:
            try:
                user_mentions.append([tweet.retweeted_status.full_text, tweet.user.screen_name, tweet.id])
            except AttributeError:  # Not a Retweet
                user_mentions.append([tweet.full_text, tweet.user.screen_name, tweet.id])
        user_mentions = pd.DataFrame(user_mentions)
        user_mentions.columns = ['text', 'user', 'tweet_ID']
        return user_mentions

    # Block users
    def block_users(self, users):
        for user in users:
            block = self.api.create_block(user)

    # Mute users
    def mute_users(self, users):
        for user in users:
            mute = self.api.create_mute(user)

    # Unblock users
    def unblock_users(self, users):
        for user in users:
            destroy_block = self.api.destroy_block(user)
