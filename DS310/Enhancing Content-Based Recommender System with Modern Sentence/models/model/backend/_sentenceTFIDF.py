import pickle
import torch
import numpy as np
from tqdm import tqdm
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from sklearn.feature_extraction.text import TfidfVectorizer

from ._base import BaseEmbedder


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

STOPWORDS = set(stopwords.words('english'))
MIN_WORDS = 4
MAX_WORDS = 200


class TFIDF_SenEmbed(BaseEmbedder):
    def __init__(self, stopwords=STOPWORDS, min_df=5, max_df=0.95, max_features=8000):
        self.min_df = min_df
        self.max_df = max_df
        self.max_features = max_features
        self.stopwords = stopwords

    def embed(self, corpus, verbose=False):
        token_stop = self.tokenizer(' '.join(STOPWORDS), lemmatize=False)
        vectorizer = TfidfVectorizer(stop_words=token_stop,
                                     tokenizer=self.tokenizer,
                                     min_df=self.min_df,
                                     max_df=self.max_df,
                                     max_features=self.max_features)
        
        embed_matrix = vectorizer.fit_transform(corpus)
        embed_matrix_dense = embed_matrix.toarray()
        embed_matrix_tensor = torch.from_numpy(embed_matrix_dense)
        self.embed_matrix = embed_matrix_tensor
        return embed_matrix_tensor
    
    # def load_embeddings_matrix(self, path):
    #     with open(path, 'rb') as f:
    #         embed_matrix = pickle.load(f)
    #         self.embed_matrix = torch.tensor(embed_matrix, dtype=float)

    def tokenizer(self, sentence, min_words=MIN_WORDS, max_words=MAX_WORDS, stopwords=STOPWORDS, lemmatize=True):
        if lemmatize:
            stemmer = WordNetLemmatizer()
            tokens = [stemmer.lemmatize(w) for w in word_tokenize(sentence)]
        else:
            tokens = [w for w in word_tokenize(sentence)]
        token = [w for w in tokens if (len(w) > min_words and len(w) < max_words
                                                            and w not in stopwords)]
        return tokens