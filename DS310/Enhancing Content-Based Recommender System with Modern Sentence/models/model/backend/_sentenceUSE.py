import numpy as np
from typing import List, Union
import torch
import tensorflow as tf
import tensorflow_hub as hub
import pickle
from tqdm import tqdm

from ._base import BaseEmbedder

class USE_SenEmbed(BaseEmbedder):
    def __init__(self, batch_size=8):
        module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
        self.model = hub.load(module_url)
        self.batch_size = batch_size

    def embed(self, corpus, verbose=False):
        input_generator = self.text_generator(corpus, self.batch_size)
        embeddings = []
        for text_batch in tqdm(input_generator, total=len(corpus)//self.batch_size):
            embeddings.append(self.model(text_batch))
        embed_matrix_tf = tf.concat(embeddings, axis=0)
        embed_matrix_np = embed_matrix_tf.numpy()
        embed_matrix = torch.from_numpy(embed_matrix_np)
        self.embed_matrix = embed_matrix
        return embed_matrix
    # def load_embeddings_matrix(self, path):
    #     with open(path, 'rb') as f:
    #         embed_matrix = pickle.load(f)
    #         self.embed_matrix = torch.tensor(embed_matrix, dtype=float)

    def text_generator(self, corpus, batch_size):
        for i in range(0, len(corpus), batch_size):
            yield corpus[i:i+batch_size]