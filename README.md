# Textual Entailment

## Summary

This project is a short exploration of textual entailment based on my projects at Georgia Tech. The experiments compare the performance of three different models:
1. A naive bayes classifier using TF-IDF vectors.
2. A neural network using embeddings generatd with BERT.
3. OpenAI's text-davinci-002 model using prompt design.

A complete write up of the results is [here]().

## Example Usage

### Text Embeddings

```sh
$ python main.py --mode preprocess --dataset dev
$ python main.py --mode embed --dataset dev
```

### Train
```sh
$ python main.py --mode train --dataset dev --model naive-bayes
$ python main.py --mode train --dataset dev --model nn --epochs 20 --batch-size 64 --learning-rate 0.001
$ python main.py --mode train --dataset dev --model text-davinci-002
```

### Evaluate
```sh
$ python main.py --mode evaluate --dataset dev --model naive-bayes --model-path 'path/to/saved_model.pkl'
$ python main.py --mode evaluate --dataset dev --model nn --model-path 'path/to/saved_model.pkl'
$ python main.py --mode evaluate --dataset dev --model text-davinci-002
```

## Model Results

| Naive Bayes   | precision | recall | f1-score | support |
| ------------- | --------- | ------ | -------- | ------- |
| contradiction | 0.51      | 0.46   | 0.48     | 3237    |
| entailment    | 0.48      | 0.58   | 0.52     | 3368    |
| neutral       | 0.52      | 0.46   | 0.49     | 3219    |
| accuracy      |           |        | 0.5      | 9824    |
| macro avg     | 0.5       | 0.5    | 0.5      | 9824    |
| weighted avg  | 0.5       | 0.5    | 0.5      | 9824    |

| Neural Network | precision | recall | f1-score | support |
| -------------- | --------- | ------ | -------- | ------- |
| contradiction  | 0.75      | 0.72   | 0.74     | 3237    |
| entailment     | 0.69      | 0.83   | 0.76     | 3368    |
| neutral        | 0.73      | 0.6    | 0.66     | 3219    |
| accuracy       |           |        | 0.72     | 9824    |
| macro avg      | 0.73      | 0.72   | 0.72     | 9824    |
| weighted avg   | 0.73      | 0.72   | 0.72     | 9824    |

| text-davinci-002 | precision | recall | f1-score | support |
| ---------------- | --------- | ------ | -------- | ------- |
| contradiction    | 0.9       | 0.42   | 0.57     | 3237    |
| entailment       | 0.6       | 0.84   | 0.7      | 3368    |
| neutral          | 0.35      | 0.39   | 0.37     | 3219    |
| accuracy         |           |        | 0.55     | 9824    |
| macro avg        | 0.62      | 0.55   | 0.55     | 9824    |
| weighted avg     | 0.62      | 0.55   | 0.55     | 9824    |

## References
1. Bowman, Samuel R., et al. "A Large Annotated Corpus for Learning Natural Language Inference." ArXiv:1508.05326 [Cs], 21 Aug. 2015, arxiv.org/abs/1508.05326.
2. Zhang, Zhuosheng, et al. "Semantics-Aware BERT for Language Understanding." ArXiv.org, 4 Feb. 2020, arxiv.org/abs/1909.02209. Accessed 15 July 2023.