import pandas as pd
import numpy as np

# Vectorizer Class
class Vectorizer:
  def __init__(self,
               pretrain_wv):
    self.pretrain_wv = pretrain_wv
    self.vector_size = pretrain_wv.vector_size

  def sentence_vector(self,
                      tokens: list) -> np.ndarray:
    """
    Generate the average Word2Vec vector for a sentence.
    """
    vectors = [self.pretrain_wv[word] for word in tokens if word in self.pretrain_wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(self.vector_size)

  def vectorize(self,
                df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorize the text column in the DataFrame.
    """
    print("Vectorizing text...")
    df["Word2Vec_Sentence_wv"] = df["Text"].apply(self.sentence_vector)
    
    return df
