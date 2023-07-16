from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import BertModel
from .SNLIFeatures import SNLIFeatures
import torch
import pandas as pd
import numpy as np

class BertEmbedder:
    def __init__(self, batch_size=32, verbose=False):
        self.batch_size = batch_size
        self.device = torch.device(
          "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.model = BertModel.from_pretrained("bert-base-uncased").to(self.device)
        self.verbose = verbose


    def transform(self, X):
        """
        Args:
            X (pandas.DataFrame): The data to use. Must have a column named
                token_ids, attention_mask, and token_type.
        """
        snli_features = self._features(X)
        dataloader = self._dataloader(snli_features)

        all_embeddings = []
        self.model.eval()

        for _, (_, tkn_ids, attn_mask, tkn_type) in enumerate(dataloader):
            input_tensor = tkn_ids.to(self.device)
            attn_mask = attn_mask.to(self.device)
            tkn_type = tkn_type.to(self.device)

            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_tensor,
                    attention_mask=attn_mask,
                    token_type_ids=tkn_type
                )

            embeddings = outputs.last_hidden_state
            embeddings = torch.mean(embeddings, dim=1)

            all_embeddings.append(embeddings.to("cpu").detach().numpy())

        embeddings_df = pd.DataFrame(np.concatenate(all_embeddings))

        return embeddings_df


    def _features(self, X):
        return SNLIFeatures(X)


    def _dataloader(self, features):
        dataloader = DataLoader(
            features, batch_size=self.batch_size, shuffle=False
        )

        if self.verbose:
            dataloader = tqdm(dataloader)

        return dataloader
