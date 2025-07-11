from datasets import Dataset
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    TrainingArguments, Trainer,
    EarlyStoppingCallback
)
import pandas as pd
import numpy as np
import torch
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Sample Dataset
data = {
    "text": [
        "Verify your account now",
        "Reset your password",
        "You've won a free iPhone",
        "Update your billing info",
        "Hello friend, long time!",
        "Let's meet at 2 PM",
        "Attached is your invoice",
        "Lunch tomorrow?"
    ],
    "label": [1, 1, 1, 1, 0, 0, 0, 0]
}
df = pd.DataFrame(data)

# 2. Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# 3. Dataset Conversion
dataset = Dataset.from_pandas(df)

def tokenize_data(example):
    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=64
    )

dataset = dataset.map(tokenize_data)
dataset = dataset.rename_column("label", "labels")
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# 4. Split Train/Test
train_test = dataset.train_test_split(test_size=0.2)
train_dataset = train_test["train"]
eval_dataset = train_test["test"]

# 5. Load BERT Model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# 6. Metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds),
        "recall": recall_score(labels, preds),
        "f1": f1_score(labels, preds),
    }

# 7. Training Configuration
training_args = TrainingArguments(
    output_dir="./bert-phishing",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    num_train_epochs=4,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    logging_dir="./logs",
    logging_steps=5,
    load_best_model_at_end=True,
    metric_for_best_model="f1"
)

# 8. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# 9. Train Model
trainer.train()

# 10. Save Final Model & Tokenizer
trainer.save_model("ai/bert_phishing_model")
tokenizer.save_pretrained("ai/bert_phishing_model")

print("✅ BERT phishing model training complete and saved to 'ai/bert_phishing_model'")
