
# DS310 Final Project: Enhancing Content-Based Recommender System with Modern Sentence Embedding Methods

![image](https://github.com/anpc849/UIT_DS310_Final_Project/assets/160831531/89ed4ba4-e8f3-44af-b63d-c82c81d24a36)






Check out `DS310_report.pdf` for detailed insights.
<br>
This repository contains the source code for DS310. You can use the following instructions to clone the repository and get started with the code.

## Prerequisites
Before you begin, make sure you have the following installed on your system. You can check the packages in the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Clone the Repository
To get started, clone this repository to your local machine using the following command:

```bash
git clone https://github.com/anpc849/UIT_DS310_Final_Project.git
```

## Set up virtual environment
It's recommended to create a virtual environment to manage project dependencies. Navigate to the project directory and run the following commands:
```bash
cd UIT_DS310_Final_Project
python -m venv venv
venv\Scripts\activate ### on windows
source venv/bin/activate ### on macOS and Linux
```
## Run the code
### Load data
``` python
from dataset.load_data import *
dataset, sentences = get_meta_data("path_to_folder") ## for embedding
test_data = pd.read_csv("path_to_csv_file") ## for evaluation recommendation task
```
### Build System
``` python
from models.system.system import SMTopicTM
system = SMTopicTM(dataset=dataset,
                       topic_model='smtopic', ## name of system
                       num_topics=5, ## num_topic of K-Means
                       dim_size=5, ## dim_size of UMAP
                       word_select_method='tfidf_idfi', ## word selection method
                       embedding='bert-base-uncased', ## name embedding
                       seed=42,
                       test_path="path_to_csv_file") ## same path of test_data
```
Note that the parameters `num_topics`, `dim_size`, and `word_select_method` are not used in the recommendation task. You should leave them as default if you are running recommendation task.

### Recommendation Task
```python
system.train_embeddings()
system.evalute_embedding_model()
```
### Topic Model Task
```python
system.train_cluster()
td_score, cv_score, npmi_score = system.evaluate_topic_model()
```
## Run gradio app
See `/webapp/gradio.ipynb` file for more detail.
<br>
See `snapshots` for  for visual demos showcasing the capabilities of my system.

## References
My source code is based on the code from the paper: [Is Neural Topic Modelling Better than Clustering? An Empirical Study on Clustering with Contextual Embeddings for Topics](https://aclanthology.org/2022.naacl-main.285.pdf)

If you use or refer to this code, please cite the following paper:
```bibtex
@inproceedings{zhang-etal-2022-neural,
    title = "Is Neural Topic Modelling Better than Clustering? An Empirical Study on Clustering with Contextual Embeddings for Topics",
    author = "Zhang, Zihan  and
      Fang, Meng  and
      Chen, Ling  and
      Namazi Rad, Mohammad Reza",
    booktitle = "Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.naacl-main.285",
    doi = "10.18653/v1/2022.naacl-main.285",
    pages = "3886--3893",
    abstract = "Recent work incorporates pre-trained word embeddings such as BERT embeddings into Neural Topic Models (NTMs), generating highly coherent topics. However, with high-quality contextualized document representations, do we really need sophisticated neural models to obtain coherent and interpretable topics? In this paper, we conduct thorough experiments showing that directly clustering high-quality sentence embeddings with an appropriate word selecting method can generate more coherent and diverse topics than NTMs, achieving also higher efficiency and simplicity.",
}
```
