
import pandas as pd
from consts import *
import torch

from transformers import BertTokenizerFast as BertTokenizer
from sentiment_tagger import SentimentTagger    

test_comments = pd.read_csv(
    DATA_PATH + '/testset/test.txt').values.tolist()


tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)


trained_model = SentimentTagger.load_from_checkpoint("./checkpoints/best-checkpoint-v17.ckpt", n_classes=80)
trained_model.eval()
trained_model.freeze()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trained_model = trained_model.to(device)


results = []
for test_comment in test_comments:
    print(test_comment)
    encoding = tokenizer.encode_plus(test_comment[0][0:512],
                                     add_special_tokens=True,
                                     max_length=512,
                                     return_token_type_ids=False,
                                     padding="max_length",
                                     return_attention_mask=True,
                                     return_tensors='pt')

    _, test_prediction = trained_model(encoding["input_ids"].to(device),
                                       encoding["attention_mask"].to(device))

    print(test_prediction.reshape(20, 4)[0])
    print(test_prediction.reshape(20, 4).argmax(dim=-1)[0])

    test_prediction = test_prediction.reshape(
        20, 4).argmax(dim=-1).flatten().cpu().numpy().tolist()
    test_prediction_list = [func(x) for x in test_prediction]

    results.append(test_comment + test_prediction_list)

    for label, prediction in zip(LABEL_COLUMNS, test_prediction_list):
        print(f"{label}: {prediction}")

results_df = pd.DataFrame(results)
results_df.columns = ["content"] + LABEL_COLUMNS
results_df.to_csv("test_results.csv")
