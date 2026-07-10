import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

training_file_path = 'file_path' # Путь к файлу с nhtybhjdjxysv датасетом.
training_text_column = 'support_response'
training_labels = ['приказной_тон', 'отказ_помочь', 'наезды_оскорбления', 'канцелярит']

class BankingDataset(Dataset):
    def __init__(self, texts, labels_data, tokenizer, max_len=128):
        self.texts = texts
        self.labels_data = labels_data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels_data[item]
        encoding = self.tokenizer(text, add_special_tokens=True, max_length=self.max_len, padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }

def compute_metrics(p):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(preds))
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= 0.5)] = 1
    return {
        'f1': f1_score(y_true=p.label_ids, y_pred=y_pred, average='micro'),
        'accuracy': accuracy_score(p.label_ids, y_pred)
    }

def calculate_human_likeness(predictions, labels_list):
    weights = np.array([1.0] * len(labels_list))
    weights = weights / np.sum(weights)
    negativity_score = np.dot(predictions, weights)
    return max(0.0, 1.0 - negativity_score)

def evaluate_csv(dataframe, model, tokenizer, labels, column_name):
    model.eval()
    def get_scores(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]
        human_score = calculate_human_likeness(probs, labels)
        res = {labels[i]: float(probs[i]) for i in range(len(labels))}
        res['HUMAN_LIKENESS_SCORE'] = float(human_score)
        return res

    results = dataframe[column_name].apply(get_scores)
    scores_df = pd.json_normalize(results)
    return pd.concat([dataframe.reset_index(drop=True), scores_df], axis=1)

df_train_full = pd.read_csv(training_file_path)
train_df, val_df = train_test_split(df_train_full, test_size=0.2, random_state=42)

model_name = "sberbank-ai/ruBert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

train_dataset = BankingDataset(train_df[training_text_column].to_numpy(), train_df[training_labels].to_numpy(), tokenizer)
val_dataset = BankingDataset(val_df[training_text_column].to_numpy(), val_df[training_labels].to_numpy(), tokenizer)

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(training_labels), problem_type="multi_label_classification")

training_args = TrainingArguments(output_dir='./results', num_train_epochs=5, per_device_train_batch_size=8, eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True, report_to="none")

trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset, compute_metrics=compute_metrics)
trainer.train()