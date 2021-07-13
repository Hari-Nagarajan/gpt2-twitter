import gpt_2_simple as gpt2
import os
from random import randint
import json
import logging
import boto3

PREFIX = "<|startoftext|>"

BLACKLIST = []


def get_base_model(model_name="355M"):
    if not os.path.isdir(os.path.join("models", model_name)):
        print(f"Downloading {model_name} model...")
        gpt2.download_gpt2(model_name=model_name)


def train(model_name="355M", tweet_combined_file_name=None, username="run1"):
    assert tweet_combined_file_name, "You must specify a file."

    get_base_model(model_name)

    sess = gpt2.start_tf_sess()
    if not os.path.exists(f"{username}_checkpoint"):
        os.makedirs(f"{username}_checkpoint")

    gpt2.finetune(
        sess,
        dataset=tweet_combined_file_name,
        run_name=username,
        checkpoint_dir=f"{username}_checkpoint",
        restore_from="latest",
        overwrite=True,
        model_name=model_name,
        steps=2000,
        sample_every=500,
        save_every=500,
    )


def load_model(username="run1"):
    global BLACKLIST

    with open("config/blacklist.json", "r") as blacklist_file:
        BLACKLIST = json.load(blacklist_file)["blacklist"]

    sess = gpt2.start_tf_sess()
    gpt2.load_gpt2(
        sess, run_name=username, checkpoint_dir=f"{username}_checkpoint",
    )
    return sess


def is_not_in_blacklist(tweet):
    """
    Checks if the tweet content contains a blacklisted word or phrase
    """
    global BLACKLIST
    if any([x in tweet.lower() for x in BLACKLIST]):
        return False
    return True


def generate_tweets(sess, username):
    """
    Generates a tweet, if the tweet is invalid (blacklist or length) then it just tries again
    """
    prefix = PREFIX
    include_prefix = False
    tweets = gpt2.generate(
        sess,
        length=100,
        temperature=0.8,
        prefix=prefix,
        truncate="<|endoftext|>",
        include_prefix=include_prefix,
        top_k=40,
        top_p=0.7,
        return_as_list=True,
        nsamples=100,
        batch_size=20,
        checkpoint_dir=f"{username}_checkpoint",
        run_name=username,
    )
    viable_tweets = []
    for tweet in tweets:
        if is_not_in_blacklist(tweet) and 280 > len(tweet) > 20:
            viable_tweets.append(tweet)
    return viable_tweets
