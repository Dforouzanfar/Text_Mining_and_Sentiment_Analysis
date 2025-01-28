from pathlib import Path
import requests
import pandas as pd

# DataManager Class
class DataManager:
  def __init__(self,
               data_path: str,
               file_name: str):
    
    self.data_path = Path(data_path)
    self.file_name = file_name
    self.raw_df = None

  def download_data(self, url: str):
    """
    Download dataset if not already present.
    """

    file_path = self.data_path / self.file_name
    if not file_path.exists():
      print("Downloading dataset...")
      with open(file_path, "wb") as f:
        request = requests.get(url)
        f.write(request.content)
      print("Dataset downloaded!")
    else:
      print("Dataset already exists.")

  def load_data(self):
    """
    Load the dataset into a pandas DataFrame.
    """

    self.raw_df = pd.read_csv(self.data_path / self.file_name)
    self.raw_df = self.raw_df[["class", "tweet"]].rename(columns={"class": "Label", "tweet": "Text"})
    
    # Remap label values
    mapping = {0: 1, 1: 2, 2: 0}
    self.raw_df["Label"] = self.raw_df["Label"].replace(mapping)
    
    # Balance the dataset by sampling equal amounts for each class
    min_sample = self.raw_df["Label"].value_counts().min()

    self.raw_df = pd.concat([
      self.raw_df[self.raw_df["Label"] == label].sample(min_sample, random_state=42)
      for label in self.raw_df["Label"].unique()
    ]).reset_index(drop=True)

    print("Data loaded and balanced successfully!")
