from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functionals as F
#Load Pre-trained Transformer
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

text = ["I love AI! This is amazing.", "I hate waiting in traffic."]

# Tokenize
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

#make prediction
with torch.no_grad():
   outputs = model(**inputs)

predictions = F.softmax(outputs.logits, dim=-1)
print(predictions)