from typing import Dict, List, Set
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, 'D:\DE_Infrastructure\coding\Tinhtoansongsong_phantan\Doan\clickstream_assessments')
from pyspark.ml.functions import predict_batch_udf
from pyspark.sql.types import FloatType
from data_processing import load_type_mappings, load_question_info, load_event_types_per_question, get_problem_qids
import torch
from joint_model import JointModel
import json
import ast
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
import pickle as pkl
import time
import plotly.graph_objects as go
from preprocess import *
# from constants import ASSISTIVE_EVENT_IDS, Correctness, Mode, TrainOptions
from data_processing import *
# from data_loading import get_sub_sequences, add_visit_level_features_and_correctness
from streamlit import runtime
from kafka import KafkaConsumer
import os
import pyspark
from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
from pyspark.sql.functions import col

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(
    page_title="Clickstream Dashboard",
    page_icon="👋",
)

with open("streamlit/unique_std.pkl", 'rb') as f:
  unique_std = pkl.load(f)

topic_name = "clickstream_data"
kafka_broker = "localhost:9092"

consumer_kafka = create_consumer(topic_name, kafka_broker)


#placeholder = st.empty()

student_dict = {}
final_event = []

Preprocesor = StreamDataProcessor()
Preprocesor.clear_variables()

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

type_mappings = load_type_mappings()
block_a_qids = {qid for _, qid in get_problem_qids("A", type_mappings)}

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

spark = SparkSession \
.builder \
.appName("spark-dl-inference-v5") \
.master("local[*]") \
.config("spark.executor.memory", "8g") \
.config("spark.driver.memory", "8g") \
.config("spark.python.worker.reuse",True) \
.config("spark.sql.execution.arrow.pyspark.enabled", True) \
.getOrCreate()



def ml_preprocess_predict(data: List[Dict]):
    test_pdf = pd.DataFrame({"id": [str(d['student_id']) for d in data],'value': [str(d) for d in data]})
    # test_pdf['value'] = test_pdf['value'].apply(ast.literal_eval)
    df = spark.createDataFrame(test_pdf)
    return df

def convert_dict_fn():
    import sys
    sys.path.insert(0, 'D:\DE_Infrastructure\coding\Tinhtoansongsong_phantan\Doan\clickstream_assessments')
    from constants import ASSISTIVE_EVENT_IDS, Correctness, Mode, TrainOptions
    from joint_model import JointModel
    def convert_dict(inputs: np.ndarray) -> np.ndarray:
        train_options = TrainOptions(dict())
        model = JointModel("lstm", True, Mode.PREDICT, type_mappings, train_options, num_labels=1, num_input_qids=len(block_a_qids))
        model.load_state_dict(torch.load("C:/Users/acer/Downloads/base_model.pt", map_location=torch.device('cpu')))
        model.eval()
        result_array = np.vectorize(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)(inputs)
        def preprocess_predict(data: dict):
            predict_loader = torch.utils.data.DataLoader(
                PerQuestionDataset_for_predict([data.copy()], block_a_qids, False, True, True, True),
                collate_fn=PerQuestionCollator_for_predict(block_a_qids),
                batch_size=1
            ) if [data] is not None else None
            test_sample = next(iter(predict_loader))
            # print(test_sample)
            return test_sample

        new_lst_dict = []
        for item in result_array:
            data = preprocess_predict(item)
            predict = model(data)
            new_lst_dict.append(predict)
        results_np = np.array(new_lst_dict, dtype=float)
        rounded_results = np.round(results_np, decimals=2)
        # results_np[results_np < 0.5] = 0
        # results_np[results_np >= 0.5] = 1
        return rounded_results
    return convert_dict


def add_visit_level_features_and_correctness(sequence: dict):
    """
    Calculate visits - for the last event of each visit to a question - the index, question id, correctness, and timestamp
    Additionally, apply visit's final correctness state to all events that are not the last in a visit
    """

    seq_len = len(sequence["question_ids"])
    visits = sequence["visits"] = {
        "idxs": [],
        "qids": [],
        "correctness": [],
        "time_deltas": [],
        "mask": None
    }
    cur_qid = sequence["question_ids"][0]
    for idx, qid in enumerate(sequence["question_ids"]):
        # Create visit for previous problem when a new question ID is encountered
        if qid != cur_qid:
            visits["idxs"].append(idx - 1)
            visits["qids"].append(cur_qid)
            visits["correctness"].append(sequence["correctness"][idx - 1])
            visits["time_deltas"].append(sequence["time_deltas"][idx - 1])
            cur_qid = qid
    # Create final visit
    visits["idxs"].append(seq_len - 1)
    visits["qids"].append(cur_qid)
    visits["correctness"].append(sequence["correctness"][seq_len - 1])
    visits["time_deltas"].append(sequence["time_deltas"][seq_len - 1])
    # Add mask for collator and model
    visits["mask"] = torch.ones(len(visits["time_deltas"]), dtype=torch.bool)

    # Assign final correctness state of visit to each event in the sequence
    start_event_idx = 0
    for visit_idx, end_event_idx in enumerate(visits["idxs"]):
        for event_idx in range(start_event_idx, end_event_idx):
            sequence["correctness"][event_idx] = visits["correctness"][visit_idx]
        start_event_idx = end_event_idx + 1

def get_sub_sequences(sequence: dict, question_ids: set, concat_visits, log_time: bool, qid_seq: bool, correctness_seq: bool):
    from constants import ASSISTIVE_EVENT_IDS, Correctness, Mode, TrainOptions

    # Split a sequence into a list of subsequences by visit
    start_idx = 0
    sub_sequences = []
    for last_idx in sequence["visits"]["idxs"]:
        qid = sequence["question_ids"][start_idx]
        if not question_ids or qid in question_ids:
            sub_seq = {
                "student_id": sequence["student_id"],
                "data_class": sequence["data_class"],
                "question_id": qid,
                "question_type": sequence["question_types"][start_idx],
                "event_types": sequence["event_types"][start_idx : last_idx + 1],
                "time_deltas": np.array(sequence["time_deltas"][start_idx : last_idx + 1]),
                "num_visits": 1,
                "max_gap": 0,
                "complete": sequence["correctness"][last_idx] != Correctness.INCOMPLETE.value,
                "correct": sequence["correctness"][last_idx] == Correctness.CORRECT.value
            }
            sub_seq["total_time"] = sub_seq["time_deltas"][-1] - sub_seq["time_deltas"][0]
            if log_time:
                # Convert to log2 as per CKT paper, add 1 to avoid log(0)
                sub_seq["time_deltas"] = np.log2(sub_seq["time_deltas"]) + 1
            if qid_seq:
                sub_seq["question_ids"] = sequence["question_ids"][start_idx : last_idx + 1]
            if correctness_seq:
                sub_seq["correctness"] = sequence["correctness"][start_idx : last_idx + 1]
            sub_sequences.append(sub_seq)
        start_idx = last_idx + 1

    # If requested, concatenate visits per qid
    if concat_visits:
        qid_to_sub_sequences = {}
        for sub_seq in sub_sequences:
            if sub_seq["question_id"] not in qid_to_sub_sequences:
                qid_to_sub_sequences[sub_seq["question_id"]] = sub_seq
            else:
                qid_sub_seqs = qid_to_sub_sequences[sub_seq["question_id"]]
                qid_sub_seqs["event_types"] += sub_seq["event_types"]
                qid_sub_seqs["max_gap"] = max(qid_sub_seqs["max_gap"], sub_seq["time_deltas"][0] - qid_sub_seqs["time_deltas"][-1])
                qid_sub_seqs["time_deltas"] = np.concatenate([qid_sub_seqs["time_deltas"], sub_seq["time_deltas"]])
                qid_sub_seqs["num_visits"] += 1
                qid_sub_seqs["total_time"] += sub_seq["total_time"]
                # Take correctness of most recent visit
                qid_sub_seqs["complete"] = sub_seq["complete"]
                qid_sub_seqs["correct"] = sub_seq["correct"]
                if qid_seq:
                    qid_sub_seqs["question_ids"] += sub_seq["question_ids"]
                if correctness_seq:
                    qid_sub_seqs["correctness"] += sub_seq["correctness"]
        sub_sequences = list(qid_to_sub_sequences.values())

    return sub_sequences

class PerQuestionDataset_for_predict(torch.utils.data.Dataset):
    def __init__(self, data: List[Dict[str, list]], allowed_qids: set, log_time: bool, qid_seq: bool, concat_visits: bool, correctness_seq: bool):
        self.data = []
        for sequence in data:
            add_visit_level_features_and_correctness(sequence)
            sub_seqs = get_sub_sequences(sequence, allowed_qids, concat_visits, log_time, qid_seq, correctness_seq)
            if concat_visits: # Ensure order with concat_visits since encodings will be fed through linear NN layer
                sub_seqs.sort(key=lambda sub_seq: sub_seq["question_id"])
            question_ids = [sub_seq["question_id"] for sub_seq in sub_seqs]

            # Insert null sub-sequences for missing questions
            if concat_visits:
                for q_idx, qid in enumerate(sorted(allowed_qids)):
                    if len(question_ids) <= q_idx or question_ids[q_idx] != qid:
                        sub_seqs.insert(q_idx, None)
                        question_ids.insert(q_idx, qid)

            self.data.append({
                "sub_seqs": sub_seqs,
                "question_ids": question_ids,
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

class PerQuestionCollator_for_predict:
    def __init__(self, available_qids: Set[int]):
        self.available_qids = available_qids
    def __call__(self, batch: List[dict]):
        # Batch info for each question
        question_batches = {
            qid: {"event_types": [], "time_deltas": [], "question_ids": [], "correctness": [], "mask": [], "target_idxs": []}
            for qid in self.available_qids
        }
        question_id_batches = []
        # labels = []

        # Each sequence in batch contains array of sub-sequences, either one per question or per visit, depending on concat_visits setting
        # Pick apart sequences to group questions across all sequences together for batch processing by encoders
        # Target index is maintained so after processing resulting encodings can be mapped back to respective sequences
        # (refers to index in unrolled and padded (sequence x sub-sequence) matrix)
        sequence_lengths = [len(seq["sub_seqs"]) for seq in batch]
        max_seq_len = max(sequence_lengths)
        for seq_idx, sequence in enumerate(batch):
            for sub_seq_idx, sub_seq in enumerate(sequence["sub_seqs"]):
                if not sub_seq: # Sub-sequence can be None when concat_visits=True to indicate the student didn't visit that question
                    continue
                question_batch = question_batches[sub_seq["question_id"]]
                question_batch["event_types"].append(torch.LongTensor(sub_seq["event_types"]))
                question_batch["time_deltas"].append(torch.from_numpy(sub_seq["time_deltas"]).type(torch.float32))
                if "question_ids" in sub_seq:
                    question_batch["question_ids"].append(torch.LongTensor(sub_seq["question_ids"]))
                if "correctness" in sub_seq:
                    question_batch["correctness"].append(torch.LongTensor(sub_seq["correctness"]))
                question_batch["mask"].append(torch.ones(len(sub_seq["event_types"]), dtype=torch.bool))
                question_batch["target_idxs"].append(seq_idx * max_seq_len + sub_seq_idx)
            question_id_batches.append(torch.LongTensor(sequence["question_ids"]))
            # labels.append(sequence["label"])

        return {
            "questions": {
                str(qid): {
                    "event_types": torch.nn.utils.rnn.pad_sequence(question_batch["event_types"], batch_first=True).to(device),
                    "time_deltas": torch.nn.utils.rnn.pad_sequence(question_batch["time_deltas"], batch_first=True).to(device),
                    "question_ids": torch.nn.utils.rnn.pad_sequence(question_batch["question_ids"], batch_first=True).to(device) if question_batch["question_ids"] else None,
                    "correctness": torch.nn.utils.rnn.pad_sequence(question_batch["correctness"], batch_first=True).to(device) if question_batch["correctness"] else None,
                    "mask": torch.nn.utils.rnn.pad_sequence(question_batch["mask"], batch_first=True).to(device),
                    "target_idxs": torch.LongTensor(question_batch["target_idxs"]).to(device),
                    "sequence_lengths": [event_types.shape[0] for event_types in question_batch["event_types"]]
                }
                for qid, question_batch in question_batches.items()
                if question_batch["event_types"] # Skip questions that had no visits at all in the batch
            },
            "question_ids": torch.nn.utils.rnn.pad_sequence(question_id_batches, batch_first=True).to(device),
            # "labels": torch.Tensor(labels).to(device),
            "sequence_lengths": sequence_lengths
        }

fig = go.Figure()
fig.update_layout(
    xaxis_title="Time spent (mins)",
    yaxis_title="Questions Ids",
    showlegend=False,
    hoverlabel=dict(
      bgcolor="#BB86FC",
      font_size=16,
      font_family='roboto',
      font_color='black'
    )

)

if 'button' not in st.session_state:
    st.session_state['button'] = False

if 'predict' not in st.session_state:
    st.session_state['predict'] = False

button = st.button("Recording")

predict = st.button("Predicting")
if predict:
  st.session_state['predict'] = True

with st.container():
  plotly_chart = st.plotly_chart(fig)

  for msg in consumer_kafka:
    record = Preprocesor.preprocess_streamdata(msg.value)
    student_dict[int(record['student_id'])] = record

    time_deltas_array = np.array(student_dict[int(record['student_id'])]['time_deltas'], dtype=float)
    question_types_name = [mapping_name_question_event(x, 'question_types') for x in student_dict[int(record['student_id'])]['question_types']]
    event_types_name = [mapping_name_question_event(x, 'event_types') for x in student_dict[int(record['student_id'])]['event_types']]
    custom_data = np.column_stack((question_types_name, event_types_name))
    ext=False
    for i, trace in enumerate(fig.data):
      if trace.name == str(record['student_id']):
        ext = True
        break
      
    if ext:
      fig.update_traces(
        x=np.round(time_deltas_array / 60, 2),
        y=student_dict[int(record['student_id'])]['question_types'],
        customdata=custom_data,
        hovertemplate='question_type: %{customdata[0]}<br>event_type: %{customdata[1]}',
        mode='lines+markers',
        #text=[record['student_id']],
        name=str(record['student_id']),
        selector=dict(name=str(record['student_id'])),
        #overwrite=True,
      )

    else:
      fig.add_trace(
        go.Scatter(
          x=np.round(time_deltas_array / 60, 2),
          y=record['question_ids'],
          customdata=custom_data,
          line=dict(color='blue'),
          hovertemplate='question_type: %{customdata[0]}<br>event_type: %{customdata[1]}',
          mode='lines+markers',
          #text=record['student_id'],  # Add student_id to text for hover
          name=str(record['student_id'])  # Add a name for legend
        )
      )


    plotly_chart.plotly_chart(fig)
    time.sleep(1)

    if button:
      st.session_state['button'] = student_dict
    if st.session_state['predict']:
      data = list(st.session_state['button'].values())
      new_data = [{k: v for k, v in item.items() if k not in ["time_start", "cur_question_id"]} for item in data]
      spark_data = ml_preprocess_predict(new_data)
      predict_fn = predict_batch_udf(convert_dict_fn,
                                   return_type=FloatType(),
                                   batch_size=10)
      result = spark_data.withColumn("pass", predict_fn("value")).toPandas()
      st.table(result[['id', 'pass']])
      st.session_state['predict'] = False
      button=False
      predict=False
      st.session_state['button']=""
      

  consumer_kafka.close()


  
