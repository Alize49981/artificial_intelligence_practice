from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf

#Load Pre-trained Transformer
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForSequenceClassification.from_pretrained(model_name)

#text = ["I love AI! This is amazing.", "I hate waiting in traffic."]

# Tokenize
inputs = tokenizer(text, return_tensors="tf", padding=True, truncation=True)

#make prediction
outputs = model(inputs)
predictions = tf.nn.softmax(outputs.logits, axis=-1)
print(predictions)