from octis.dataset.dataset import Dataset
import pandas as pd

def get_meta_data(path):
    dataset = Dataset()
    dataset.load_custom_dataset_from_folder(path)
    token_lists = dataset.get_corpus()
    sentences = [' '.join(text_list) for text_list in token_lists]
    
    return dataset, sentences


def get_interaction_data(path):
    dataset = pd.read_csv(path)
    return dataset

