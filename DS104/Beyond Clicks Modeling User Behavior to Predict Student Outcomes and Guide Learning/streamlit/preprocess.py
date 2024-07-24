import json
import re
from typing import Dict, List, Set
import pandas
#from clickstream-assessments/constants import ASSISTIVE_EVENT_IDS, Correctness
from collections import Counter
#from clickstream-assessments/data_processing import *
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, 'D:\DE_Infrastructure\coding\Tinhtoansongsong_phantan\Doan\clickstream_assessments')
#from constants import ASSISTIVE_EVENT_IDS, Correctness
from data_processing import *
import copy
import streamlit.components.v1 as components
from kafka import KafkaConsumer
import logging
import plotly.graph_objects as go
import torch
#from data_loading import get_sub_sequences, add_visit_level_features_and_correctness
from data_processing import load_type_mappings, load_question_info, load_event_types_per_question, get_problem_qids
import training
#from constants import Mode, TrainOptions
#from joint_model import JointModel


type_mappings = load_type_mappings()
qa_key = load_question_info()



def create_consumer(topic_name, kafka_broker):
    try:
        consumer = KafkaConsumer(
        topic_name,
        bootstrap_servers=kafka_broker,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        value_deserializer=lambda x: json.loads(x.decode("utf-8")))
    except Exception as e:
        logging.exception("Couldn't create the consumer")
        consumer = None
    return consumer

class StreamDataProcessor:
    def __init__(self):
        self.cur_question_id = None
        #self.start_time = None
        self.final_event = []
        self.student_to_sequences: Dict[int, dict] = {}
        self.student_to_qa_states: Dict[int, Dict[str, QuestionAnswerState]] = {}
        self.student_to_q_visits: Dict[int, Dict[str, List[List[float]]]] = {}

    def preprocess_streamdata(self, event):
        event = json.loads(event)
        event = pd.Series(event)   
        event['EventTime'] = pd.to_datetime(event['EventTime'])
        
        block = event["Block"]

        ## xem lại sequence, nó đang áp dụng cho tất cả student_id
        sequence: Dict[str, list] = self.student_to_sequences.setdefault(int(event["STUDENTID"]), {
                "data_class": None,
                "student_id": int(event["STUDENTID"]),
                "question_ids": [],
                "question_types": [],
                "time_start": None,
                'cur_question_id': None,
                "event_types": [],
                "time_deltas": [],
                "correctness": [],
                "assistive_uses": {eid: 0 for eid in ASSISTIVE_EVENT_IDS},
                "q_stats": {},
                "block_a_score": 0,
                "block_b_score": 0
            })
        
        if event["Observable"] == "Click Progress Navigator":
            return copy.deepcopy(sequence)

        if not event['EventTime']:
            return copy.deepcopy(sequence)

        if not sequence["event_types"]:
            sequence['time_start'] = event["EventTime"]
        time_delta = (event["EventTime"] - sequence['time_start']).total_seconds()

        qid = event["AccessionNumber"]
        eid = type_mappings["event_types"][event["Observable"]]

        if eid in ASSISTIVE_EVENT_IDS:
            sequence["assistive_uses"][eid] += 1

        qa_states = self.student_to_qa_states.setdefault(event["STUDENTID"], {
            qid: QuestionAnswerState(qid, qa_key) for qid in type_mappings["question_ids"]
        })
        qa_state = qa_states[qid]
        qa_state.process_event(event)

        if event["STUDENTID"] not in self.student_to_q_visits:
            #self.cur_question_id = None
            sequence['cur_question_id'] = None

        qid_to_visits = self.student_to_q_visits.setdefault(event["STUDENTID"], {
            qid: [] for qid in type_mappings["question_ids"]
        })
        q_visits = qid_to_visits[qid]
        #if qid != self.cur_question_id: # If we went to a new question, start a new visit
        if qid != sequence['cur_question_id']:
            q_visits.append([time_delta, time_delta])
            #self.cur_question_id = qid
            sequence['cur_question_id'] = qid
        else: # Update current visit
            q_visits[-1][1] = time_delta

        sequence["question_ids"].append(type_mappings["question_ids"][qid])
        sequence["question_types"].append(type_mappings["question_types"][event["ItemType"]])
        sequence["event_types"].append(eid)
        sequence["time_deltas"].append(time_delta)
        sequence["correctness"].append(qa_state.get_correctness_label().value)

        # ... (rest of your existing code)

        result = copy.deepcopy(sequence)
        return result
    
    def clear_variables(self):
        self.final_event = []
        self.student_to_sequences = {}
        self.student_to_qa_states = {}
        self.student_to_q_visits = {}
        self.cur_question_id = None
        #self.start_time = None

# final_event = []
# student_to_sequences: Dict[int, dict] = {}
# student_to_qa_states: Dict[int, Dict[str, QuestionAnswerState]] = {}
# student_to_q_visits: Dict[int, Dict[str, List[List[float]]]] = {}
# cur_question_id = None
# start_time = None
def preprocess_streamdata(event):
    global cur_question_id, start_time, final_event

    #header = ["STUDENTID", "Block", "AccessionNumber", "ItemType","Observable", "ExtendedInfo", "EventTime"]
    #event = event.split(',')
    #event = pd.Series(event, index = header)

    event = json.loads(event)
    event = pd.Series(event)    
    event['EventTime'] = pd.to_datetime(event['EventTime'])
    
    block = event["Block"]

    if event["Observable"] == "Click Progress Navigator":
        return

    if not event['EventTime']:
        return

    sequence: Dict[str, list] = student_to_sequences.setdefault(event["STUDENTID"], {
            "data_class": None,
            "student_id": event["STUDENTID"],
            "question_ids": [],
            "question_types": [],
            "event_types": [],
            "time_deltas": [],
            "correctness": [],
            "assistive_uses": {eid: 0 for eid in ASSISTIVE_EVENT_IDS},
            "q_stats": {},
            "block_a_score": 0,
            "block_b_score": 0
        })

    if not sequence["event_types"]:
        start_time = event["EventTime"]
    time_delta = (event["EventTime"] - start_time).total_seconds()

    qid = event["AccessionNumber"]
    eid = type_mappings["event_types"][event["Observable"]]

    if eid in ASSISTIVE_EVENT_IDS:
        sequence["assistive_uses"][eid] += 1

    qa_states = student_to_qa_states.setdefault(event["STUDENTID"], {
        qid: QuestionAnswerState(qid, qa_key) for qid in type_mappings["question_ids"]
    })
    qa_state = qa_states[qid]
    qa_state.process_event(event)

    if event["STUDENTID"] not in student_to_q_visits:
        cur_question_id = None

    qid_to_visits = student_to_q_visits.setdefault(event["STUDENTID"], {
        qid: [] for qid in type_mappings["question_ids"]
    })
    q_visits = qid_to_visits[qid]
    if qid != cur_question_id: # If we went to a new question, start a new visit
        q_visits.append([time_delta, time_delta])
        cur_question_id = qid
    else: # Update current visit
        q_visits[-1][1] = time_delta

    sequence["question_ids"].append(type_mappings["question_ids"][qid])
    sequence["question_types"].append(type_mappings["question_types"][event["ItemType"]])
    sequence["event_types"].append(eid)
    sequence["time_deltas"].append(time_delta)
    sequence["correctness"].append(qa_state.get_correctness_label().value)

    # Final processing based on per-student question data
    qids_to_track = [qid for qid, q_info in qa_key.items() if (not block or q_info["block"] == block) and q_info["answer"] != "na"]
    for student_id, seq in student_to_sequences.items():
        q_stats = seq["q_stats"]
        for qid in qids_to_track:
            q_visits = student_to_q_visits[student_id][qid]
            qa_state = student_to_qa_states[student_id][qid]
            q_correctness = qa_state.get_correctness_label()
            q_stats[qid] = {
                "time": sum(end_time - start_time for start_time, end_time in q_visits),
                "visits": len(q_visits),
                "correct": q_correctness.value,
                "final_state": qa_state.get_state_string()
            }
            if q_correctness == Correctness.CORRECT:
                q_block = qa_key[qid]["block"]
                if q_block == "A":
                    seq["block_a_score"] += 1
                elif q_block == "B":
                    seq["block_b_score"] += 1
    # if event['STUDENTID'] != sequence['student_id']:
    #final_event += list(student_to_sequences.values())
    result = copy.deepcopy(sequence)
    return result


def process_assitive_uses(student):
  assistive_uses = student['assistive_uses']
  assistive_uses_name_mapping = {
    4: 'Open Calculator',
    8: 'Eliminate Choice',
    17: 'Scratchwork Mode On',
    24: 'Highlight',
    27: 'TextToSpeech',
    34: 'Increase Zoom'
  }
  assistive_uses = {assistive_uses_name_mapping[key]: value for key, value in assistive_uses.items()}
  return assistive_uses

def mapping_name_question_event(ids_q, type_opt):
  mapping_name = {
    'question_types': {
      0: 'Directions',
      1: 'MCSS',
      2: 'MatchMS ',
      3: 'MultipleFillInBlank',
      4: 'FillInBlank',
      5: 'BlockReview',
      6: 'CompositeCR',
      7: 'TimeLeftMessage',
      8: 'GridMS',
      9: 'ZonesMS',
      10: 'BQMCSS',
      11: 'Help',
      12: 'TimeOutMessage',
      13: 'Dialog'},
    'event_types': {
      0: 'Enter Item',
      1: 'Next',
      2: 'Exit Item',
      3: 'Click Choice',
      4: 'Open Calculator',
      5: 'Move Calculator',
      6: 'Calculator Buffer',
      7: 'DropChoice',
      8: 'Eliminate Choice',
      9: 'Vertical Item Scroll',
      10: 'Receive Focus',
      11: 'Math Keypress',
      12: 'Lose Focus',
      13: 'Close Calculator',
      14: 'Click Progress Navigator',
      15: 'First Text Change',
      16: 'Yes',
      17: 'Leave Section',
      18: 'Open Equation Editor',
      19: 'Equation Editor Button',
      20: 'Close Equation Editor',
      21: 'Scratchwork Mode On',
      22: 'Scratchwork Mode Off',
      23: 'Draw',
      24: 'Clear Answer',
      25: 'Scratchwork Erase Mode On',
      26: 'Erase',
      27: 'Scratchwork Draw Mode On',
      28: 'Clear Scratchwork',
      29: 'Last Text Change',
      30: 'Change Theme',
      31: 'TextToSpeech',
      32: 'Increase Zoom',
      33: 'Horizontal Item Scroll',
      34: 'Decrease Zoom',
      35: 'Scratchwork Highlight Mode On',
      36: 'Highlight',
      37: 'Back',
      38: 'OK',
      39: 'Hide Timer',
      40: 'Show Timer',
      41: 'No'}}
  ## create a mapping_ids_name from mapping_name
  name = mapping_name[type_opt][ids_q]
  return name



def ColourWidgetText(wgt_txt, wch_colour = '#000000'):
    htmlstr = """<script>var elements = window.parent.document.querySelectorAll('*'), i;
                    for (i = 0; i < elements.length; ++i) { if (elements[i].innerText == |wgt_txt|) 
                        elements[i].style.color = ' """ + wch_colour + """ '; } </script>  """

    htmlstr = htmlstr.replace('|wgt_txt|', "'" + wgt_txt + "'")
    components.html(f"{htmlstr}", height=0, width=0)


# def add_std_to_trace(fig, record):
#     time_deltas_array = np.array(record['time_deltas'], dtype=float)
#     question_types_name = [mapping_name_question_event(x, 'question_types') for x in record['question_types']]
#     event_types_name = [mapping_name_question_event(x, 'event_types') for x in record['event_types']]
#     custom_data = np.column_stack((question_types_name, event_types_name))

#     # Check if the student_id is already in the plot
#     ext = False
#     for i, trace in enumerate(fig.data):
#         if trace.name == str(record['student_id']):
#             ext = True
#             break
#     #existing_trace_indices = [i for i, trace in enumerate(fig.data) if trace.name == str(record['student_id'])]

#     if ext:
#         # If it exists, update the existing trace
#         fig.update_traces(
#             x=[np.round(time_deltas_array / 60, 2)],
#             y=[record['question_ids']],
#             customdata=[custom_data],
#             hovertemplate='question_type: %{customdata[0]}<br>event_type: %{customdata[1]}',
#             mode='lines+markers',
#             #text=[record['student_id']],
#             name=str(record['student_id']),
#             selector=dict(name=str(record['student_id'])),
#             overwrite=True,
#         )
#     else:
#         # If it doesn't exist, add a new trace
#         fig.add_trace(
#             go.Scatter(
#                 x=np.round(time_deltas_array / 60, 2),
#                 y=record['question_ids'],
#                 customdata=custom_data,
#                 line=dict(color='blue'),
#                 hovertemplate='question_type: %{customdata[0]}<br>event_type: %{customdata[1]}',
#                 mode='lines+markers',
#                 #text=record['student_id'],  # Add student_id to text for hover
#                 name=str(record['student_id'])  # Add a name for legend
#             )
#         )

# def clear_variables():
#     global final_event, student_to_sequences, student_to_qa_states, student_to_q_visits, cur_question_id, start_time
#     final_event = []
#     student_to_sequences = {}
#     student_to_qa_states = {}
#     student_to_q_visits = {}
#     cur_question_id = None
#     start_time = None