
import torch

from transformers import BertTokenizerFast as BertTokenizer
from model.sentiment_tagger import SentimentTagger

BERT_MODEL_NAME = 'bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

trained_model = SentimentTagger.load_from_checkpoint("./checkpoint/best-checkpoint-v17.ckpt", n_classes=80)
trained_model.eval()
trained_model.freeze()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trained_model = trained_model.to(device)

torch.save(trained_model,"./checkpoint/model.pt")
