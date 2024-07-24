from ._base import BaseEmbedder
from ._sentencetransformers import SentenceTransformerBackend
from ._sentenceTFIDF import TFIDF_SenEmbed
from ._sentenceUSE import USE_SenEmbed
from ._sentencew2v import Word2Vec_SenEmbed

def select_backend(embedding_model):
  
    if isinstance(embedding_model, str):
        if "tfidf" in embedding_model:
            return TFIDF_SenEmbed()
        elif "use" in embedding_model:
            return USE_SenEmbed()
        elif 'word2vec' in embedding_model:
            return Word2Vec_SenEmbed()
        else:
            return SentenceTransformerBackend(embedding_model)
    else:
        raise ValueError("The embedding model must be a string")
    
