import gpt_2_simple as gpt2
import tweepy
from random import randint
import json
import logging

POST_ON_TWITTER = True
PREFIXES = ["Joke:", "Fact:", "<|startoftext|>", "LOL", "Idea:"]

BLACKLIST = []

logger = logging.getLogger(__name__)


def is_not_in_blacklist(tweet):
    """
    Checks if the tweet content contains a blacklisted word or phrase
    """
    global BLACKLIST
    if any([x in tweet.lower() for x in BLACKLIST]):
        return False
    return True


def generate_tweet(sess):
    """
    Generates a tweet, if the tweet is invalid (blacklist or length) then it just tries again
    """
    prefix = PREFIXES[randint(0, len(PREFIXES) - 1)]
    tweet = gpt2.generate(
        sess, length=200, temperature=0.7, prefix=prefix, truncate="<|endoftext|>", include_prefix=False, top_k=40, return_as_list=True,
    )[0].strip()
    if is_not_in_blacklist(tweet) and len(tweet) < 140:
        return tweet
    else:
        return generate_tweet(sess)


def get_twitter_api(twitter_conf):
    """
    Just builds the twitter api object
    """
    try:
        auth = tweepy.OAuthHandler(twitter_conf["consumer_key"], twitter_conf["consumer_secret"])
        auth.set_access_token(twitter_conf["access_token"], twitter_conf["access_token_secret"])
    except KeyError as e:
        logger.error("twitter.json is improperly formatted.")
        raise e

    return tweepy.API(auth)

def main():
    global BLACKLIST

    with open("config/blacklist.json", "r") as blacklist_file:
        BLACKLIST = json.load(blacklist_file)["blacklist"]

    with open("config/twitter.json", "r") as twitter_conf_file:
        twitter_conf = json.load(twitter_conf_file)

    api = get_twitter_api(twitter_conf)
    sess = gpt2.start_tf_sess()
    gpt2.load_gpt2(sess)

    tweet = generate_tweet(sess)
    print(tweet)

    if POST_ON_TWITTER:
        api.update_status(status=tweet)


if __name__ == "__main__":
    main()
