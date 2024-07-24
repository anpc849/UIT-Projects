import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
import pickle as pkl
import time
import numpy as np
import plotly.graph_objects as go
from preprocess import *
import sys
sys.path.insert(0, 'D:\DE_Infrastructure\coding\Tinhtoansongsong_phantan\Doan\clickstream_assessments')
from constants import ASSISTIVE_EVENT_IDS, Correctness
from data_processing import *
#from data_processing import StreamDataProcessor
from streamlit import runtime
from kafka import KafkaConsumer

# type_mappings = load_type_mappings()
# qa_key = load_question_info()

# final_event = []
# student_to_sequences: Dict[int, dict] = {}
# student_to_qa_states: Dict[int, Dict[str, QuestionAnswerState]] = {}
# student_to_q_visits: Dict[int, Dict[str, List[List[float]]]] = {}
# cur_question_id = None
# start_time = None



st.set_page_config(
    page_title="Real-Time ClickStream Education Exam",
    page_icon="✅",
    layout="wide",
)

st.title("Real-Time ClickStream Education Exam")

#st.write(qa_key)

# clickstream_data = {2333000033: None, 2333000955: None}
# with open('/content/student_to_sequences_studentA_full.pkl', 'rb') as f:
#   history_data_a = pkl.load(f)
# with open('/content/student_to_sequences_studentB_full.pkl', 'rb') as f:
#   history_data_b = pkl.load(f)
# history_dict = {2333000033: history_data_a, 2333000955: history_data_b}

with open("streamlit/unique_std.pkl", 'rb') as f:
  unique_std = pkl.load(f)


topic_name = "clickstream_data"
kafka_broker = "localhost:9092"

consumer_kafka = create_consumer(topic_name, kafka_broker)
Preprocesor = StreamDataProcessor()
student_dict = {}
final_event = []

student_filter = st.selectbox("Select the student_id", unique_std)
Preprocesor.clear_variables()
#clear_variables()
placeholder = st.empty()


for msg in consumer_kafka:
  record = Preprocesor.preprocess_streamdata(msg.value)
  try:
    student_dict[int(record['student_id'])] = record
    # aaa = json.loads(msg.value)
    # if int(aaa['STUDENTID']) == int(student_filter):
    #   st.write(msg.value)
  except:
    st.write(f"Data of student: <span style='color: red;'>{student_filter}</span> is not available", unsafe_allow_html=True)
    continue
  ##st.write(student_dict[student_filter])

  with placeholder.container():
    time_exam, unique_question, total_score = st.columns(3)
    try:
      value_time_exam = round(student_dict[student_filter]['time_deltas'][-1] / 60, 2)
      
      #value_time_exam = round(student_dict[student_filter]['time_deltas'][-1], 2)

      value_unique_question = len(set(student_dict[student_filter]['question_ids']))
      value_total_score = student_dict[student_filter]['block_a_score']

      time_exam.metric(
        label = 'Time spent on the exam (mins)',
        value = value_time_exam,
      )
      color = '#90EE91' if value_time_exam < 10 else ('#FFEB00' if value_time_exam < 20 else '#FF496D')
      #color = '#90EE91' if value_time_exam < 10*60 else ('#FFEB00' if value_time_exam < 20*60 else '#FF496D')
      ColourWidgetText(str(value_time_exam), str(color))

      unique_question.metric(
        label = 'Encountered questions',
        value = value_unique_question
      )

      total_score.metric(
        label = 'Total Score',
        value = value_total_score
      )

      st.markdown("---")
      columns = st.columns([2, 1])
      with columns[0]:
        time_deltas_array = np.array(student_dict[student_filter]['time_deltas'], dtype=float)
        question_types_name = [mapping_name_question_event(x, 'question_types') for x in student_dict[student_filter]['question_types']]
        event_types_name = [mapping_name_question_event(x, 'event_types') for x in student_dict[student_filter]['event_types']]
        fig = px.line(
          #x=np.round(time_deltas_array / 60, 2),
          x=np.round(time_deltas_array, 2),
          y=student_dict[student_filter]['question_ids'],
          hover_data={
            'question_type': question_types_name, 
            'event_type': event_types_name
            },
          markers=True,
          range_x=[0, None],
          )

        fig.update_layout(
          xaxis_title="Time spent (seconds)",
          yaxis_title="Questions IDs",
          showlegend=False,
          hoverlabel=dict(
          bgcolor="#BB86FC",
          font_size=16,
          font_family="Rockwell",
          font_color="black"
          )
        )

        # Display the plot using Streamlit
        st.plotly_chart(fig)

      # Right column for the assistive tool information
      with columns[1]:
        st.markdown("""
          <p style='text-align:center;font-size:25px;'><strong>Assistive tool used</strong></p>
          """, unsafe_allow_html=True
        )
        assistive_data = process_assitive_uses(student_dict[student_filter])
        for key, value in assistive_data.items():
          st.markdown(f"<div style='text-align:center;'>{key}: {value}</div>", unsafe_allow_html=True)

    except:
      st.write(f"Data of student: <span style='color: red;'>{student_filter}</span> is not available", unsafe_allow_html=True)
      continue

  time.sleep(1)
  #st.write(student_dict[student_filter])



consumer_kafka.close()