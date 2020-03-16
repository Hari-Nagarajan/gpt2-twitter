import boto3


def send_to_queue(queue, tweet):
    queue.send_message(MessageBody=tweet)


def get_queue(username=None):
    assert username, "You must specify a username."

    sqs = boto3.resource("sqs", region_name="us-east-1")
    queue = sqs.get_queue_by_name(QueueName=f"TweetQueue-{username}")
    return queue
