import sys
import asyncio
from kafka_producer import send_video_to_kafka, create_producer

def main(hashtag, topic_name):
    kafka_broker = "localhost:9092"
    num_videos = 3
    producer = create_producer(kafka_broker)
    if producer is None:
        print("Error")

    send_video_to_kafka(hashtag_name=hashtag, producer=producer, topic_name=topic_name, kafka_broker=kafka_broker, num_videos=num_videos)

    producer.flush()

    producer.close()
    
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python test_kafka.py <hashtag> <topic_name>")
        sys.exit(1)
    hashtag = sys.argv[1]
    topic_name = sys.argv[2]
    main(hashtag, topic_name)
