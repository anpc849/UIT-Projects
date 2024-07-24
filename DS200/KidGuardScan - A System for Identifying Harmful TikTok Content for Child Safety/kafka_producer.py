from TikTokApi import TikTokApi
import asyncio
import os
from kafka import KafkaProducer
import json
import logging
import time 

ms_token = "YIsVKMt1xAF8isDwchtjtn734ktfYiGn-ocIEqW4atO2IQDIqx3G_abmUOjalmyxKHXlrGtimEVSqox3Z_FVFaLm7d9FVwBB_P7pICqmLWw44sr3EKYLKVZaj3YkeWFQDJp2K1WDoF6qGMg2Ng=="

context_options = {
    'viewport': {'width': 1280, 'height': 1024},
    'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.150 Safari/537.36'
}


async def get_hashtag_videos(hashtag_name, num_videos):
    videos = []
    async with TikTokApi() as api:
        await api.create_sessions(ms_tokens=[ms_token], num_sessions=1, sleep_after=3, context_options=context_options)
        tag = api.hashtag(name=hashtag_name)
        async for video in tag.videos(count=num_videos):
            videos.append(video)
    return videos

async def get_info_video(url):
    async with TikTokApi() as api:
        await api.create_sessions(ms_tokens=[ms_token], num_sessions=1, sleep_after=3, context_options=context_options)
        video = api.video(url=url)
        video_info = await video.info()
    return video_info



def create_producer(kafka_broker):
    try:
        producer = KafkaProducer(
            bootstrap_servers=kafka_broker, 
            value_serializer=lambda x:json.dumps(x).encode('utf-8'),
            key_serializer=lambda x:json.dumps(x).encode('utf-8'),
        )
    except Exception as e:
        logging.exception("Couldn't create the producer")
        producer = None
    
    return producer

def send_video_to_kafka(hashtag_name, producer, topic_name, kafka_broker, num_videos):
    videos = asyncio.run(get_hashtag_videos(hashtag_name, num_videos))
    for video in videos[:num_videos]:
        infor_dict = video.as_dict
        url_video = "https://www.tiktok.com/@" + infor_dict['author']['uniqueId'] + "/video/" + infor_dict['id']
        video_infor = asyncio.run(get_info_video(url_video))
        producer.send(topic_name, value=video_infor, key=hashtag_name)
        time.sleep(2)
    producer.flush()

# async def send_videos_to_kafka(hashtag_name, producer, topic_name, kafka_broker, num_videos):
#     videos = await get_hashtag_videos(hashtag_name, num_videos)
#     for video in videos:
#         producer.send(topic_name, value=video.as_dict, key=hashtag_name)
#         time.sleep(2)
#     producer.flush()

# def send_video_to_kafka(hashtag_name, producer, topic_name, kafka_broker, num_videos):
#     asyncio.run(send_videos_to_kafka(hashtag_name, producer, topic_name, kafka_broker, num_videos))



