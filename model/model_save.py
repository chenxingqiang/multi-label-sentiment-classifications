
import torch
import torch.nn as nn

from transformers import BertTokenizerFast as BertTokenizer
from sentiment_tagger import SentimentTagger

import pytorch_lightning as pl

BERT_MODEL_NAME = 'bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)




trained_model = SentimentTagger.load_from_checkpoint(
    "/Users/xingjian/PyProjects/Multi-Emotion-SGM/checkpoint/best-checkpoint-v17.ckpt", n_classes=80)
trained_model.eval()
trained_model.freeze()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trained_model = trained_model.to(device)

torch.save(trained_model,"./checkpoint/model.pt")
