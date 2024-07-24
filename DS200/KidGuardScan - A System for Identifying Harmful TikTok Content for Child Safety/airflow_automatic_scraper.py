import uuid
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
import json
from kafka import KafkaProducer
import time
import logging
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from webdriver_manager.microsoft import EdgeChromiumDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from TikTokApi import TikTokApi
import asyncio
import os

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
            value_serializer=lambda x: json.dumps(x).encode('utf-8'),
            key_serializer=lambda x: json.dumps(x).encode('utf-8'),
        )
    except Exception as e:
        logging.exception("Couldn't create the producer")
        producer = None
    return producer

def send_video_to_kafka(hashtag_name, producer, topic_name, num_videos):
    videos = asyncio.run(get_hashtag_videos(hashtag_name, num_videos))
    for video in videos[:num_videos]:
        infor_dict = video.as_dict
        url_video = "https://www.tiktok.com/@" + infor_dict['author']['uniqueId'] + "/video/" + infor_dict['id']
        video_infor = asyncio.run(get_info_video(url_video))
        producer.send(topic_name, value=video_infor, key=hashtag_name)
        time.sleep(2)
    producer.flush()

def send_data(hashtag=None, topic_name=None):
    kafka_broker = ["host.docker.internal:9092"]
    num_videos = 3
    producer = create_producer(kafka_broker)
    if producer is None:
        logging.error("Error creating Kafka producer")
        return

    send_video_to_kafka(hashtag_name=hashtag, producer=producer, topic_name=topic_name, num_videos=num_videos)
    producer.flush()
    producer.close()

default_args = {
    'owner': 'airscholar',
    'start_date': datetime(2023, 9, 3, 10, 0)
}

def create_webdriver(url):
    opt = Options()
    opt.add_argument('--disable-dev-shm-usage')
    opt.add_argument("--lang=vi-VN")
    opt.add_argument(f"--window-size=1366,768")
    opt.add_argument(f'--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.182 Safari/537.36')
    opt.add_argument('--disable-blink-features=AutomationControlled')
    opt.add_argument("--disable-extensions")
    opt.add_argument("--proxy-bypass-list=*")
    opt.add_argument('--ignore-certificate-errors')
    opt.add_argument("--password-store=basic")
    opt.add_argument("--no-sandbox")
    opt.add_argument("--disable-dev-shm-usage")
    opt.add_argument("--disable-extensions")
    opt.add_argument("--enable-automation")
    opt.add_argument("--disable-browser-side-navigation")
    opt.add_argument("--disable-web-security")
    opt.add_argument("--disable-dev-shm-usage")
    opt.add_argument("--disable-infobars")
    opt.add_argument("--disable-gpu")
    opt.add_argument("--disable-setuid-sandbox")
    opt.add_argument("--disable-software-rasterizer")

    driver = webdriver.Remote(
        command_executor='http://selenium:4444/wd/hub',
        options=opt
    )
    driver.get(url)
    return driver

def stream_data():
    driver = create_webdriver("https://ads.tiktok.com/business/creativecenter/inspiration/popular/hashtag/pc/vi")
    sended_hashtag_path = "/opt/airflow/sended_hashtag.txt"
    with open(sended_hashtag_path, 'r') as file:
        sended_hashtag = file.read().splitlines()
    len_hashtag = 0
    while len_hashtag <= 3:
        wait = WebDriverWait(driver, 20)
        cards = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "a.CardPc_container___oNb0")))
        for card in cards:
            hashtag = card.find_element(By.CSS_SELECTOR, "span.CardPc_titleText__RYOWo").text.strip().replace("#", "")
            if hashtag not in sended_hashtag:
                len_hashtag += 1
                send_data(hashtag, "kafka_topic")
                with open(sended_hashtag_path, 'a') as file:
                    file.write(hashtag + "\n")
            else:
                continue
            
        wait = WebDriverWait(driver, 10)
        view_more_button = wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, "div.ViewMoreBtn_viewMoreBtn__fOkv2[data-testid='cc_contentArea_viewmore_btn']")))
        actions = ActionChains(driver)
        actions.move_to_element(view_more_button).perform()
        view_more_button.click()

    driver.close()
    driver.quit()

with DAG('user_automation',
         default_args=default_args,
         schedule_interval='@daily',
         catchup=False) as dag:

    streaming_task = PythonOperator(
        task_id='stream_data_from_api',
        python_callable=stream_data
    )
