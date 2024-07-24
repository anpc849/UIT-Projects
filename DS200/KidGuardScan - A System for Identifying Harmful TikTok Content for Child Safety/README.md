# KidGuardScan: A System for Identifying Harmful TikTok Content for Child Safety

## Introduction
TikTok's widespread use among children raises concerns due to potentially harmful content. KidGuardScan addresses this by employing a two-phase approach. In the offline phase, a custom dataset trains a video classification model. This model is integrated into the online phase, where Apache Kafka and Apache Spark handle real-time video streaming and classification. The system analyzes streamed videos to detect and classify harmful content, promoting a safer online environment for children.

See the overall system in `images/overall_system.png`

*A Demo of KidGuardScan*

[Demo KidGuardScan Sytem](https://github.com/user-attachments/assets/58a970bb-4c70-4c9b-a0a0-7adcbf27c1de)


*Performance comparison of video classification models on the TikHarm Dataset.*

| Model        | Time inference         | Clip Duration         | Precision | Recall | F1 score|
| -------------|:----------------------:|:---------------------:|:---------:|:------:|:-------:|
| VideoMAE     | 31.3s                  | 1.13s                 | 0.8646  | 0.8633  | 0.8625   |
| VideoMAE     | **28.74s**                 | 2.13s                 | 0.8648  | 0.8646  | 0.8638   |
| TimeSformer  | 31.43s                 | 1.13s                 | 0.8712  | 0.8709  | 0.8700   |
| TimeSformer  | 32.08s                 | 2.13s                 | **0.8758**  | **0.8747**  | **0.8739**   |





## Installation and Getting Started

### TikHarm Dataset & Checkpoint Model

TikHarm Dataset is public here: https://www.kaggle.com/datasets/anhoangvo/tikharm-dataset

Download the best model with the following commands:
```bash
!pip install -q gdown
!gdown 1ZgKn2b1-d-wFhFtzA20bjEDtz33nE-TC
!unzip checkpoint-2300.zip
```
### Spark Streaming
Initialize the Spark cluster and start listening to streaming data using `consumer.ipynb`. Preprocess and send metavideos with predicted labels to the MongoDB database.

### Kafka Producer
Send TikTok metavideos to the Kafka topic using `producer.ipynb`, where preprocessing is handled by Spark streaming.

### Web Application
Run the script below to interact with the database and display analysis:
```python
streamlit run webapp/overall.py
```

### Training Model Notebook Tutorial
Refer to the [Kaggle Notebook](https://www.kaggle.com/code/anhoangvo/how-to-use-hugging-face-for-fine-tuning-on-the-tik) for a detailed tutorial.
