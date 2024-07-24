import pickle
import torch
import numpy as np
from tqdm import tqdm
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

from gensim.models import KeyedVectors
from gensim.models.word2vec import Word2Vec

# from sklearn.feature_extraction.text import TfidfVectorizer

from ._base import BaseEmbedder


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

STOPWORDS = set(stopwords.words('english'))
MIN_WORDS = 4
MAX_WORDS = 200


class Word2Vec_SenEmbed(BaseEmbedder):
    def __init__(self, stopwords=STOPWORDS, min_df=5, max_df=0.95, max_features=8000, word2vec_model=None, path=None):
        self.min_df = min_df
        self.max_df = max_df
        self.max_features = max_features
        self.stopwords = stopwords
        # if not word2vec_model:
        #     self.word2vec = word2vec_model
        # else:
        #     self.load_embeddings_matrix(path)
        EMBEDDING_FILE = '/content/drive/MyDrive/DS310_Final/GoogleNews-vectors-negative300.bin.gz'
        self.google_word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)
    
    def embed(self, corpus, verbose=False):
        token_stop = self.tokenizer(' '.join(STOPWORDS), lemmatize=False)
        corpus = [self.tokenizer(sentence) for sentence in corpus]
        embedding_matrices = []

        if verbose:
            corpus = tqdm(corpus, desc="Embedding Progress")

        for sentence in corpus:
            embedding_matrix = self.sentence_embedding(sentence, self.google_word2vec)
            embedding_matrices.append(embedding_matrix)

        embedding_matrices = np.array(embedding_matrices)
        embed_matrix_tensor = torch.from_numpy(embedding_matrices)
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
    

    def sentence_embedding(self, sentence, word2vec_model):
    # Calculate the mean of word vectors for words present in the Word2Vec model
        return np.mean([word2vec_model[word] for word in sentence if word in word2vec_model], axis=0)