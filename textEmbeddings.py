from transformers import BertModel, BertTokenizer
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
text = ""
inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
outputs = model(**inputs)
embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()


