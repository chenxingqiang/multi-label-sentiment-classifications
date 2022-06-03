
import numpy as np

from tqdm.auto import tqdm
import torch
from transformers import BertTokenizerFast as BertTokenizer
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy, auroc
from sklearn.metrics import classification_report

from model.sentiment_tagger import SentimentTagger
from model.sentiment_dataset import SentimentDataset
from consts import *


pl.seed_everything(RANDOM_SEED)


test_df = read_xy(
    DATA_PATH + '/validationset/sentiment_analysis_validationset.csv')

tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

trained_model = SentimentTagger.load_from_checkpoint(
    "./checkpoint/best-checkpoint-v17.ckpt", n_classes=80)
trained_model.eval()
trained_model.freeze()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trained_model = trained_model.to(device)

val_dataset = SentimentDataset(
    test_df,
    tokenizer,
    max_token_len=MAX_TOKEN_COUNT
)

predictions = []
labels = []

for item in tqdm(val_dataset):
    _, prediction = trained_model(
        item["input_ids"].unsqueeze(dim=0).to(device),
        item["attention_mask"].unsqueeze(dim=0).to(device)
    )
    predictions.append(prediction.flatten())
    labels.append(item["labels"].int().reshape(80))

predictions = torch.stack(predictions).detach().cpu()
labels = torch.stack(labels).detach().cpu()

print("accuracy:", accuracy(predictions, labels, threshold=THRESHOLD))
print("predictions:", predictions.shape)
print("labels:", labels.shape)
print("AUROC per tag")

for i, name in enumerate(LABEL_COLUMNS_ALL):
    tag_auroc = auroc(predictions[:, i], labels[:, i], pos_label=1)
    print(f"{name}: {tag_auroc}")

y_pred = predictions.numpy()
y_true = labels.numpy()

upper, lower = 1, 0
y_pred = np.where(y_pred > THRESHOLD, upper, lower)

print(classification_report(
    y_true,
    y_pred,
    target_names=LABEL_COLUMNS_ALL,
    zero_division=0
))
