# GPT-2 Twitter 

This repo allows you to scrape tweets from a list of twitter accounts, use them to fine-tune a GPT-2 Model and then generate tweets. I am currently running this on my twitter account. All the tweets with 'KNSR' as the source are generated with this.

https://twitter.com/HariRVA

## How to run

See config directory for config instructions

Run all commands from the root of the project. This assumes you have python 3.6+ and pipenv installed. 
#### Install dependencies
```
pipenv install
```

### Scrape Tweets
```
Usage: twirclejerk.py scrape [OPTIONS] USERNAME
```


### Train model with your scraped tweets
```
Usage: twirclejerk.py train [OPTIONS] USERNAME

Options:
  --base_model TEXT  Base model to use
  --help             Show this message and exit.

```


### Generate a tweet
```
Usage: twirclejerk.py generate [OPTIONS] USERNAME
```