import pandas as pd
import numpy as np
from tqdm.auto import tqdm

import torch
import pytorch_lightning as pl
from transformers import BertTokenizerFast as BertTokenizer
from pytorch_lightning.metrics.functional import accuracy, auroc
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from model.sentiment_dataset import SentimentDataModule,SentimentDataset
from model.sentiment_tagger import SentimentTagger
from consts import  * 
from consts import read_xy

pl.seed_everything(RANDOM_SEED)


df = read_xy(DATA_PATH + '/trainingset/sentiment_analysis_trainingset.csv')
test_df = read_xy(
    DATA_PATH + '/validationset/sentiment_analysis_validationset.csv')

train_df, val_df = train_test_split(df, test_size=0.2)

tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

data_module = SentimentDataModule(
    train_df,
    val_df,
    tokenizer,
    batch_size=BATCH_SIZE,
    max_token_len=MAX_TOKEN_COUNT
)

steps_per_epoch = len(train_df) // BATCH_SIZE
total_training_steps = steps_per_epoch * N_EPOCHS

warmup_steps = total_training_steps // 5
warmup_steps, total_training_steps

model = SentimentTagger(
    n_classes=len(LABEL_COLUMNS)*4,
    n_warmup_steps=warmup_steps,
    n_training_steps=total_training_steps)


checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints",
    filename="best-checkpoint",
    save_top_k=1,
    verbose=True,
    monitor="val_loss",
    mode="min"
)

logger = TensorBoardLogger("lightning_logs", name="sentiment-comments")
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=2)

trainer = pl.Trainer(
    logger=logger,
    checkpoint_callback=checkpoint_callback,
    callbacks=[early_stopping_callback],
    max_epochs=N_EPOCHS,
    progress_bar_refresh_rate=30,
    gpus=1,
)

trainer.fit(model, data_module)
trainer.test()

trained_model = SentimentTagger.load_from_checkpoint(
    trainer.checkpoint_callback.best_model_path,
    n_classes=len(LABEL_COLUMNS_ALL)
)
trained_model.eval()
trained_model.freeze()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trained_model = trained_model.to(device)

val_dataset = SentimentDataset(
    val_df,
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
