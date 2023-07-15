"""
Main file for the project.
"""
import os
import argparse
from dotenv import load_dotenv
from datetime import datetime
import json
from tqdm import tqdm
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from src import (
    TextualEntailmentPrompter,
    BertEmbedder,
    BertTextPreprocessor,
    NeuralNetworkClassifier
)


DATASETS = {
    "dev": {
        "snli": "snli_1.0_dev.txt",
        "features": "snli_dev_features.csv",
        "embeddings": "snli_dev_embeddings.csv",
        "texts": "snli_dev_texts.csv",
        "tfidf": "snli_dev_tfidf.csv"
    },
    "train": {
        "snli": "snli_1.0_train.txt",
        "features": "snli_train_features.csv",
        "embeddings": "snli_train_embeddings.csv",
        "texts": "snli_train_texts.csv",
        "tfidf": "snli_train_tfidf.csv"
    },
    "test": {
        "snli": "snli_1.0_test.txt",
        "features": "snli_test_features.csv",
        "embeddings": "snli_test_embeddings.csv",
        "texts": "snli_test_texts.csv",
        "tfidf": "snli_test_tfidf.csv"
    }
}


def load_embeddings(data_dir, dataset):
    embeddings_path = os.path.join(
        data_dir, DATASETS[dataset]['embeddings']
    )
    if not os.path.exists(embeddings_path):
        return torch.tensor([])

    embeddings_df = pd.read_csv(embeddings_path, header=None)

    return torch.tensor(embeddings_df.values, dtype=torch.float32)


def load_labels(data_dir, dataset):
    labels_path = os.path.join(data_dir, DATASETS[dataset]['features'])
    if not os.path.exists(labels_path):
        return torch.tensor([])

    y = pd.read_csv(labels_path, sep="|", header=0)['gold_label']
    y = LabelEncoder().fit_transform(y.values)
    y = torch.tensor(y, dtype=torch.int)

    return y


def load_texts(data_dir, dataset):
    data_path = os.path.join(data_dir, DATASETS[dataset]['snli'])
    if not os.path.exists(data_path):
        return pd.DataFrame([])

    data = pd.read_csv(data_path, sep="\t", header=0)
    data = data[data['gold_label'] != "-"]
    data = data[['sentence1', 'sentence2', 'gold_label']]
    data = data.dropna().reset_index(drop=True)

    return data


def make_pipeline(model):
    pipe = Pipeline(
        [
            ('tfidf', TfidfVectorizer(
                    max_features=2000, stop_words='english', analyzer='word'
                )
            ),
            ('train', model)
        ],
        verbose=5
    )
    return pipe


def save_model(model, model_name, data_dir):
    out_path = os.path.join(
        data_dir, f"{model_name}_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.pkl"
    )
    with open(out_path, 'wb') as file:
        pickle.dump(model, file)

    return out_path


def save_results(model, dataset, y_test, y_pred):
    results_df = pd.DataFrame()
    results_df['y_test'] = y_test
    results_df['y_pred'] = y_pred
    results_path = os.path.join(
        "./results",
        f'{model}_{dataset}_results_{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.csv'
    )
    results_df.to_csv(results_path, header=True, index=False)

    return results_path


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default="help",
        help="mode to run the program in"
    )
    parser.add_argument(
        "-dir",
        "--data-dir",
        type=str,
        default="data",
        help="Path to the data directory."
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="dev",
        help="Dataset to use. Assumes that the dataset \
              is in the ./data directory."
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs to train the model for."
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate for the model."
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for the model."
    )
    parser.add_argument(
        "-ml",
        "--model",
        type=str,
        default="nn",
        help="Model to use for classification."
    )
    parser.add_argument(
        "-mp",
        "--model-path",
        type=str,
        default="",
        help="Path to the model to use for classification."
    )

    args = parser.parse_args()

    if args.mode == "evaluate" and args.model == "text-davinci-002":
        texts = load_texts(args.data_dir, args.dataset)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        result_path = f"{args.model}_results_{timestamp}.json"

        for idx, sample in enumerate(tqdm(texts.values)):
            result = {}
            result['sentence1'] = sample[0]
            result['sentence2'] = sample[1]
            result['gold_label'] = sample[2]

            predicted_label = TextualEntailmentPrompter(
                os.environ.get("OPENAI_API_KEY")
            ).predict(
                result['sentence1'],
                result['sentence2']
            )
            result['predicted_label'] = predicted_label

            with open(result_path, 'a') as f:
                json.dump(result, f)
                f.write('\n')
    elif args.mode == "preprocess":
        file_path = os.path.join(args.data_dir, DATASETS[args.dataset]['snli'])
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist.")
            exit(1)

        data = pd.read_csv(file_path, sep="\t", header=0)

        # NOTE: remove samples with no label and NaN values
        data = data[data['gold_label'] != '-']
        data = data[['sentence1', 'sentence2', 'gold_label']]
        data = data.dropna().reset_index(drop=True)

        features_df = BertTextPreprocessor().transform(data)

        out_path = os.path.join(args.data_dir, DATASETS[args.dataset]['features'])
        features_df.to_csv(out_path, header=True, sep="|", index=False)

        print(f"Saved preprocessed data for {args.dataset} to {out_path}.")
    elif args.mode == "embed":
        file_path = os.path.join(args.data_dir, DATASETS[args.dataset]['features'])
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist.")
            exit(1)

        features_df = pd.read_csv(file_path, sep="|", header=0)

        # NOTE: convert string representations of lists to lists
        features_df['token_ids'] = features_df['token_ids'].apply(eval)
        features_df['attention_mask'] = features_df['attention_mask'].apply(eval)
        features_df['token_type'] = features_df['token_type'].apply(eval)

        embeddings_df = BertEmbedder(verbose=True).transform(features_df)

        out_path = os.path.join(args.data_dir, DATASETS[args.dataset]['embeddings'])
        embeddings_df.to_csv(
            out_path,
            header=False,
            index=False
        )

        print(f"Saved embeddings for {args.dataset} to {out_path}.")
    elif args.mode == "train" and args.model in ['ada', 'nbayes']:
        texts = load_texts(args.data_dir, args.dataset)
        if texts.empty:
            print(f"No sentence pairs for {args.dataset} found in {args.data_dir}.")
            exit(1)

        X_train = texts['sentence1'] + " " + texts['sentence2']
        y_train = texts['gold_label']

        if args.model == "nbayes":
            pipe = make_pipeline(MultinomialNB(alpha=4.0, force_alpha=False))
        elif args.model == "ada":
            pipe = make_pipeline(AdaBoostClassifier(n_estimators=300))
        else:
            print(f"Model {args.model} not supported.")
            exit(1)

        pipe.fit(X_train, y_train)

        out_path = save_model(pipe, args.model, args.data_dir)
        print(f"Saved {args.model} pipeline to {out_path}.")
    elif args.mode == "train" and args.model == "nn":
        X_train = load_embeddings(args.data_dir, args.dataset)
        if X_train.numel() == 0:
            print(f"No embeddings for {args.dataset} found in {args.data_dir}.")
            exit(1)

        y_train = load_labels(args.data_dir, args.dataset)
        if y_train.numel() == 0:
            print(f"No labels for {args.dataset} found in {args.data_dir}.")
            exit(1)

        print("Training neural network classifier...")

        nn_clf = NeuralNetworkClassifier(
            in_features=768,
            out_features=100,
            n_classes=3,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            verbose=True
        )
        nn_clf.fit(X_train, y_train)

        out_path = save_model(nn_clf, args.model, args.data_dir)
        print(f'Saved model to {out_path}.')
    elif args.mode == "evaluate" and args.model == "nn":
        X_test = load_embeddings(args.data_dir, args.dataset)
        if X_test.numel() == 0:
            print(f"No embeddings for {args.dataset} found in {args.data_dir}.")
            exit(1)

        y_test = load_labels(args.data_dir, args.dataset)
        if y_test.numel() == 0:
            print(f"No labels for {args.dataset} found in {args.data_dir}.")
            exit(1)

        if args.model_path is None:
            print("Please specify a path to a trained model.")
            exit(1)

        with open(args.model_path, 'rb') as file:
            model = pickle.load(file)

        y_pred = model.predict(X_test)

        print(classification_report(y_test, y_pred, zero_division=0))

        results_path = save_results(args.model, args.dataset, y_test, y_pred)
        print(f"Saved results to {results_path}.")
    elif args.mode == "evaluate" and args.model in ['ada', 'nbayes']:
        texts = load_texts(args.data_dir, args.dataset)
        if texts.empty:
            print(f"No sentence pairs for {args.dataset} found in {args.data_dir}.")
            exit(1)

        X_test = texts['sentence1'] + " " + texts['sentence2']
        y_test = texts['gold_label']

        if args.model_path is None:
            print("Please specify a path to a trained model.")
            exit(1)

        with open(args.model_path, 'rb') as file:
            pipe = pickle.load(file)

        y_pred = pipe.predict(X_test)

        print(classification_report(y_test, y_pred, zero_division=0))

        results_path = save_results(args.model, args.dataset, y_test, y_pred)
        print(f"Saved results to {results_path}.")
    else:
        parser.print_help()
        exit(0)