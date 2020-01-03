import gpt_2_simple as gpt2
import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import csv
import glob

MIN_LENGTH = 50
MAX_LENGTH = 200
STEP_LENGTH = 50
TWEET_PATH = "tweets"
MODEL_NAME = "355M"
TWEET_COMBINED_FILE_NAME = "combined.csv"


def get_base_model(model_name="355M"):
    if not os.path.isdir(os.path.join("models", model_name)):
        print(f"Downloading {model_name} model...")
        gpt2.download_gpt2(model_name=model_name)


if __name__ == "__main__":
    get_base_model(MODEL_NAME)

    sess = gpt2.start_tf_sess()
    csv_files = glob.glob(os.path.join(TWEET_PATH, '*.csv'))

    tweet_sets = [f for f in csv_files if isfile(f)]
    combined_csv = pd.concat([pd.read_csv(f) for f in tweet_sets])
    combined_csv.to_csv(TWEET_COMBINED_FILE_NAME, index=False, encoding="utf-8")

    gpt2.finetune(
        sess,
        dataset=TWEET_COMBINED_FILE_NAME,
        restore_from="latest",
        overwrite=True,
        model_name=MODEL_NAME,
        steps=2000,
        sample_every=500,
        save_every=500,
    )