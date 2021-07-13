import click


@click.group()
def cli():
    pass


@click.command()
@click.argument("username")
def scrape(username):
    from utils.twitter import scrape_all_follows

    click.echo(f"Scraping the list of following for {username}")
    scrape_all_follows(username)
    click.echo(f"Scraped all tweets for the list of following for {username}")


@click.command()
@click.option("--base_model", default="355M", help="Base model to use")
@click.argument("username")
def train(base_model, username):
    from utils.model import train

    click.echo(f"Training model for {username}")
    train(
        model_name=base_model,
        tweet_combined_file_name=f"{username}/{username}_combined.csv",
        username=username,
    )


@click.command()
@click.argument("username")
def generate(username):
    from utils.model import generate_tweets, load_model

    click.echo(f"Loading model for {username}")
    sess = load_model(username=username)

    tweets = generate_tweets(sess)
    click.echo(tweets[0])


@click.command()
@click.argument("username")
def generate_to_sqs(username):
    from utils.model import generate_tweets, load_model
    from utils.sqs import get_queue, send_to_queue

    click.echo(f"Preparing queue for {username}")
    queue = get_queue(username=username)

    click.echo(f"Loading model for {username}")
    sess = load_model(username=username)

    for x in range(100):
        tweets = generate_tweets(sess, username)
        click.echo(f"Got {len(tweets)} more tweets!")
        for tweet in tweets:
            send_to_queue(queue, tweet)


cli.add_command(scrape)
cli.add_command(train)
cli.add_command(generate)
cli.add_command(generate_to_sqs)


if __name__ == "__main__":
    cli()
