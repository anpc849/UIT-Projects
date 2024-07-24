import streamlit as st
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import pandas as pd
import time
from datetime import datetime
import matplotlib.pyplot as plt
from streamlit_echarts import st_echarts
from streamlit_echarts import st_pyecharts
import plotly.express as px
from wordcloud import WordCloud
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import plotly.graph_objects as go
import random
import string

## Introduction layer
#empty_space = "&nbsp;"
st.markdown(f"<h2 style='text-align: center;'>Hệ thống phân loại video TikTok có chứa nội dung độc hại cho trẻ em theo thời gian thực</h1>", unsafe_allow_html=True)
course, group = st.columns([4,1])
with course:
    st.markdown("##### Đồ án môn DS200.O21")
with group:
    st.markdown("##### Nhóm số 7")

st.image('/teamspace/studios/this_studio/KidGuardScan/images/pikaso_texttoimage_35mm-film-photography-An-illustration-of-the-AIdri.jpeg', caption="Illustration created by AI")
num_videos, stats_labels = st.columns(2)


if 'database' not in st.session_state:

    ## Mongo DB
    with open('KidGuardScan/uri_mongodb.txt', 'r') as file:
        st.session_state.uri = file.read().strip()

    st.session_state.client = MongoClient(st.session_state.uri, server_api=ServerApi('1'))
    st.session_state.db = st.session_state.client['VideoTikTok']
    st.session_state.collection = st.session_state.db['videos']

    st.session_state.num_videos = num_videos
    st.session_state.stats_labels = stats_labels

    # Send a ping to confirm a successful connection
    try:
        st.session_state.client.admin.command('ping')
        print("Pinged your deployment. You successfully connected to MongoDB!")
    except Exception as e:
        print(e)

# Function to fetch data from MongoDB
def fetch_data():
    data = list(st.session_state.collection.find())
    df = pd.DataFrame(data)
    df['created_at'] = pd.to_datetime(df['created_at'].astype(int), unit='s')
    df = df.sort_values(by='created_at')
    return df, len(data)


colors = {
        "Safe": ['#00cc00'],  # Blue
        "Adult Content": ['#ffcc99'],  # Red
        "Harmful Content": ['#ff33ff'],  # Orange
        "Suicide": ['#c50a0a']  # Red
    }


# Function to compute statistics
def compute_statistics(df):
    total_videos = len(df)
    label_counts = (df['predicted_label'].value_counts(normalize=True) * 100).round(2)
    data = [{"value": count, "name": label, "itemStyle": {"color": colors.get(label, ['#ffcc99'])[0]}} for label, count in zip(label_counts.index, label_counts.values)]

    options = {
    "tooltip": {"trigger": "item"},
    # "legend": {"top": "5%", "left": "center"},
    "series": [
        {
            "type": "pie",
            "radius": ["40%", "70%"],
            "avoidLabelOverlap": True,
            "itemStyle": {
                "borderRadius": 10,
                "borderColor": "#fff",
                "borderWidth": 2,
            },
            "label": {"show": None, "position": "center"},
            "emphasis": {
                "label": {"show": None, "fontSize": "10", "fontWeight": "bold"}
            },
            "labelLine": {"show": None},
            "data": data,
            "tooltip": {"textStyle": {"fontSize": 10}}
            }
        ],
    }

    return total_videos, options

stop_database = st.button("Stop")
if stop_database:
    st.session_state.client.close()
    st.stop()

# Streamlit app layout
st.title("Real-Time TikTok Video Statistics")
# st.sidebar.header("Settings")
refresh_rate = 30


# Real-time data display
placeholder = st.empty()
chart_holder = st.empty()
word_cloud_chart = st.empty()
pie_chart = st.empty()

colors_line = {
    "Safe": '#00cc00',  # Green
    "Adult Content": '#ffcc99',  # Light Peach
    "Harmful Content": '#ff33ff',  # Pink
    "Suicide": '#c50a0a'  # Dark Red
}
def generate_random_key(length=8):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

current_len = 0
while True:
    data, new_len = fetch_data()
    if (data.empty) or (current_len == new_len):
        st.write("No new data")
        time.sleep(30)
        continue
    


    current_len = new_len

    random_key = generate_random_key()
    
    total_videos, label_percentages = compute_statistics(data)

    # Prepare for plotting
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))

    with placeholder.container():

        num_videos, stats_labels = st.columns(2)

        num_videos.metric(
            label = 'Preprocessed Videos',
            value = total_videos
        )
        with stats_labels.container():
            st_echarts(
                    options=label_percentages, height="100px", key=random_key
            )
        # st.write(data[['user_id', 'description', 'predicted_label', 'created_at']].tail(10))


    with chart_holder.container():
        df_grouped = data.groupby([data['created_at'].dt.to_period('M'), 'predicted_label']).size().reset_index(name='count_video')
        df_grouped['created_at'] = df_grouped['created_at'].dt.to_timestamp()
        # Plotting with Plotly
        fig = px.line(
            df_grouped, 
            x='created_at', 
            y='count_video', 
            color='predicted_label', 
            title='Video Count Trends by Label Over Time',
            labels={'count_video': 'Count of Videos', 'created_at': 'Date', 'predicted_label': 'Label'},
            line_shape='linear',
            markers=True,
            color_discrete_map=colors_line,
        )

        # Update layout for better readability
        fig.update_layout(
            margin_r=100,
            xaxis_title='Date',
            yaxis_title='Count of Videos',
            legend_title='Predicted Label'
        )

        # Display the chart
        st.plotly_chart(fig)



    with word_cloud_chart.container():
        # Get the unique labels
        labels = data['predicted_label'].unique()

        # Create a subplot with 2 rows and 2 columns
        fig = make_subplots(rows=2, cols=2, subplot_titles=[f'Word Cloud for {label}' for label in labels])

        # Extract hashtags and generate word clouds
        data['hashtags_list'] = data['hashtags'].str.split(" ")
        hashtags_by_label = data.groupby('predicted_label')['hashtags_list'].apply(lambda x: [item for sublist in x for item in sublist])

        for i, label in enumerate(labels):
            text = ' '.join(hashtags_by_label[label])
            wordcloud = WordCloud(
                width=400,  # Adjust width and height as needed
                height=200,
                background_color='white',
            ).generate(text)

            # Convert word cloud to image array
            wordcloud_image = wordcloud.to_array()

            # Add the word cloud image to the subplot
            fig.add_trace(
                go.Image(z=wordcloud_image),
                row=(i // 2) + 1, col=(i % 2) + 1
            )

        # Update layout for the figure
        fig.update_layout(
            height=600,  # Adjust the height to fit the 2x2 grid
            showlegend=False,
            title_text="Word Clouds Analysis",
        )

        # Update axes to be off for all subplots
        fig.update_xaxes(showticklabels=False, showgrid=False)
        fig.update_yaxes(showticklabels=False, showgrid=False)

        # Show the figure
        st.plotly_chart(fig)

    time.sleep(15)