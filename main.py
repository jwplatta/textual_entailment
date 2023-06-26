"""
Main file for the project.
"""
import os
import argparse
from dotenv import load_dotenv
import json
import tqdm

from src import TextualEntailmentPrompter


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
        "-d",
        "--data",
        type=str,
        default="snli_1.0_dev.jsonl",
        help="Dataset to use. Assumes that the dataset \
              is in the data/snli_1.0 directory."
    )

    args = parser.parse_args()

    if args.mode == "prompt":
        file_path = os.path.join("data/snli_1.0", args.data)
        data = [json.loads(row) for row in open(file_path, 'r', encoding='utf-8')]
        for sample in tqdm.tqdm(data):
            result = {}
            result['gold_label'] = sample['gold_label']
            result['sentence1'] = sample['sentence1']
            result['sentence2'] = sample['sentence2']
            predicted_label = TextualEntailmentPrompter(
                os.environ.get("OPENAI_API_KEY")
            ).predict(
                sample['sentence1'],
                sample['sentence2']
            )
            result['predicted_label'] = predicted_label

            with open('results.json', 'a') as f:
                json.dump(result, f)
                f.write('\n')
    else:
        parser.print_help()