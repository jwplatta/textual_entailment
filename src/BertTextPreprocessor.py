from transformers import BertTokenizer

class BertTextPreprocessor:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.max_input_length = self.tokenizer \
            .max_model_input_sizes['bert-base-uncased']


    def fit(self, X, y=None):
        return X

    def transform(self, X):
        """
        Args:
            X (pandas.DataFrame): Dataframe with columns 'sentence1' and 'sentence2'.
        """
        # NOTE: add special tokens
        X['sentence1'] = self.tokenizer.cls_token + \
            X['sentence1'] + self.tokenizer.sep_token
        X['sentence2'] = X['sentence2'] + self.tokenizer.sep_token

        # NOTE: tokenize
        X['sent1_tokens'] = X['sentence1'].apply(self.tokenizer.tokenize)
        X['sent2_tokens'] = X['sentence2'].apply(self.tokenizer.tokenize)

        # NOTE: make input type ids
        X['sent1_type_ids'] = X['sent1_tokens'].apply(lambda tokens: [0] * len(tokens))
        X['sent2_type_ids'] = X['sent2_tokens'].apply(lambda tokens: [1] * len(tokens))

        # NOTE: get token ids
        X['sequence'] = X['sent1_tokens'] + X['sent2_tokens']
        X['token_ids'] = X['sequence'].apply(self.tokenize)

        # NOTE: create the attention mask
        X['attention_mask'] = X['sequence'].apply(
            lambda seq: [1] * len(seq) + [0] * (128 - len(seq))
        )

        # NOTE: pad the token type ids
        X['token_type'] = X['sent1_type_ids'] + X['sent2_type_ids']
        X['token_type'] = X['token_type'].apply(
            lambda token_type_ids: token_type_ids + [0] * (128 - len(token_type_ids))
        )
        X['token_count'] = X['sequence'].apply(len)

        return X


    def max_input_tokens(self, sentence):
        tokens = sentence.strip().split(" ")
        tokens = tokens[:self.max_input_length]
        return tokens

    def tokenize(self, seq):
        return self.tokenizer.encode(
            seq, max_length=128, truncation=True, padding="max_length"
        )