from torch.utils.data import  Dataset
import torch

class SNLIFeatures(Dataset):
    def __init__(self, data):
        """
        Args:
            data (pandas.DataFrame): The data to use.
        """
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        labels = self.data.loc[index]['gold_label']
        token_ids = self.data.loc[index]['token_ids']
        attention_mask = self.data.loc[index]['attention_mask']
        token_type = self.data.loc[index]['token_type']

        return labels, \
          torch.tensor(token_ids), \
          torch.tensor(attention_mask), \
          torch.tensor(token_type)