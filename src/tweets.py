import twint
import re
import csv
from tqdm import tqdm
import logging
from datetime import datetime
from time import sleep
import os
import json
import sys
from pathlib import Path

# Surpress random twint warnings
logger = logging.getLogger()
logger.disabled = True


def is_reply(tweet):
    """
    Determines if the tweet is a reply to another tweet.
    Requires somewhat hacky heuristics since not included w/ twint
    """

    # If not a reply to another user, there will only be 1 entry in reply_to
    if len(tweet.reply_to) == 1:
        return False

    # Check to see if any of the other users "replied" are in the tweet text
    users = tweet.reply_to[1:]
    conversations = [user["username"] in tweet.tweet for user in users]

    # If any if the usernames are not present in text, then it must be a reply
    if sum(conversations) < len(users):
        return True
    return False


def update_resume_file(tweet_id):
    """
    Writes the latest tweet id to a temp file so the scrape can resume.
    """
    with open(".temp", "w", encoding="utf-8") as f:
        f.write(str(tweet_id))


def download_tweets(username=None, strip_usertags=False, strip_hashtags=False):
    """Download public Tweets from a given Twitter account
    into a format suitable for training with AI text generation tools.
    :param username: Twitter @ username to gather tweets.
    :param limit: # of tweets to gather; None for all tweets.
    :param include_replies: Whether to include replies to other tweets.
    :param strip_usertags: Whether to remove user tags from the tweets.
    :param strip_hashtags: Whether to remove hashtags from the tweets.
    """

    assert username, "You must specify a username to download tweets from."

    pattern = r"http\S+|pic\.\S+|\xa0|…"

    if strip_usertags:
        pattern += r"|@[a-zA-Z0-9_]+"

    if strip_hashtags:
        pattern += r"|#[a-zA-Z0-9_]+"

    update_resume_file(-1)

    filename = f"tweets/{username}_tweets.csv"
    file_exists = os.path.isfile(filename)

    with open(filename, "a", encoding="utf8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tweets"])
        if not file_exists:
            w.writeheader()  # file doesn't exist yet, write a header
        path = Path(filename)
        last_modified = datetime.fromtimestamp(path.stat().st_mtime).date()

        c_lookup = twint.Config()
        c_lookup.Username = username
        c_lookup.Store_object = True
        c_lookup.Hide_output = True
        if datetime.now().date() > last_modified:
            c_lookup.Since = last_modified.strftime("%Y-%m-%d")

        twint.run.Lookup(c_lookup)
        limit = twint.output.users_list[-1].tweets
        print(f"Retrieving up to {limit} tweets for @{username}...")

        for i in range((limit // 20) - 1):
            tweet_data = []

            # twint may fail; give it up to 5 tries to return tweets
            for _ in range(0, 4):
                if len(tweet_data) == 0:
                    c = twint.Config()
                    c.Store_object = True
                    c.Hide_output = True
                    c.Username = username
                    c.Limit = 40
                    c.Retweets = False
                    c.Min_likes = 5
                    c.Media = False
                    c.Videos = False
                    c.Images = False
                    c.Links = "exclude"
                    c.Resume = ".temp"
                    if datetime.now().date() > last_modified:
                        c.Since = last_modified.strftime("%Y-%m-%d")

                    c.Store_object_tweets_list = tweet_data

                    twint.run.Search(c)

                    # If it fails, sleep before retry.
                    if len(tweet_data) == 0:
                        sleep(1.0)
                else:
                    continue

            # If still no tweets after multiple tries, we're done
            if len(tweet_data) == 0:
                break

            if i > 0:
                tweet_data = tweet_data[20:]

            tweets = [re.sub(pattern, "", tweet.tweet).strip() for tweet in tweet_data if not is_reply(tweet)]

            for tweet in tweets:
                if tweet != "" and len(tweet) > 20 and "/" not in tweet:
                    w.writerow({"tweets": tweet.strip().encode("utf-8", errors="ignore").decode("utf-8", errors="replace")})
    os.remove(".temp")


if __name__ == "__main__":
    with open("config/accounts.json", "r") as accounts_file:
        accounts = json.load(accounts_file)
        for account in accounts["accounts"]:
            download_tweets(username=account)
