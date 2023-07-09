import matplotlib.pyplot as plt
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

RESULTS = {
    'nbayes': 'results/nbayes_test_results_2023-07-05 13:52:25.csv',
    'nn': 'results/nn_test_results_2023-07-05 13:52:50.csv',
    'text-davinci-002': 'results/text-davinci-002_test_results_2023-07-05 14:08:34.json'
}

test_dataset_path = './data/snli_1.0_test.txt'
test_df = pd.read_csv(test_dataset_path, sep='\t', header=0)

nbayes_df = pd.read_csv(RESULTS['nbayes'], header=0)

labels = ['neutral', 'entailment', 'contradiction']
labeler = LabelEncoder().fit(labels)
nn_df = pd.read_csv(RESULTS['nn'], header=0)
nn_df['y_pred'] = labeler.inverse_transform(nn_df['y_pred'])
nn_df['y_test'] = labeler.inverse_transform(nn_df['y_test'])

davinci = [
  json.loads(row) for row in open(RESULTS['text-davinci-002'], 'r', encoding='utf-8')
]
davinci_df = pd.DataFrame(data=davinci)
davinci_df = davinci_df[['gold_label', 'predicted_label']]


# Classifiaction reports
nbayes_cr = classification_report(nbayes_df['y_test'], nbayes_df['y_pred'])
nn_cr = classification_report(nn_df['y_test'], nn_df['y_pred'])
davinci_cr = classification_report(
  davinci_df['gold_label'],
  davinci_df['predicted_label']
)

with open('results/classification_reports.txt', 'w') as f:
    f.write('Naive Bayes\n')
    f.write(nbayes_cr)
    f.write('\n\nNeural Network\n')
    f.write(nn_cr)
    f.write('\n\nDavinci\n')
    f.write(davinci_cr)

# Confusion matrices
nbayes_cm = confusion_matrix(
  nbayes_df['y_test'],
  nbayes_df['y_pred'],
  labels=labels
)
nn_cm = confusion_matrix(
  nn_df['y_test'],
  nn_df['y_pred'],
  labels=labels
)
davinci_cm = confusion_matrix(
  davinci_df['gold_label'],
  davinci_df['predicted_label'],
  labels=labels
)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(nbayes_cm, cmap='Greens')
axes[0].set_title('Naive Bayes')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')
axes[0].set_xticks(np.arange(3))
axes[0].set_xticklabels(labels)
axes[0].set_yticks(np.arange(3))
axes[0].set_yticklabels(labels)
axes[0].tick_params(axis='both', which='both', length=0)

for i in range(nbayes_cm.shape[0]):
    for j in range(nbayes_cm.shape[1]):
        axes[0].text(j, i, nbayes_cm[i, j], ha='center', va='center', color='black')

axes[1].imshow(nn_cm, cmap='Reds')
axes[1].set_title('Neural Network')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')
axes[1].set_xticks(np.arange(3))
axes[1].set_xticklabels(labels)
axes[1].set_yticks(np.arange(3))
axes[1].set_yticklabels(labels)

for i in range(nn_cm.shape[0]):
    for j in range(nn_cm.shape[1]):
        axes[1].text(j, i, nn_cm[i, j], ha='center', va='center', color='black')

axes[2].imshow(davinci_cm, cmap='Blues')
axes[2].set_title('Davinci')
axes[2].set_xlabel('Predicted')
axes[2].set_ylabel('Actual')
axes[2].set_xticks(np.arange(3))
axes[2].set_xticklabels(labels)
axes[2].set_yticks(np.arange(3))
axes[2].set_yticklabels(labels)

for i in range(davinci_cm.shape[0]):
    for j in range(davinci_cm.shape[1]):
        axes[2].text(j, i, davinci_cm[i, j], ha='center', va='center', color='black')

plt.tight_layout()
plt.savefig('results/confusion_matrices.png')
plt.show()