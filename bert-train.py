import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import pickle
from sklearn.utils import resample
def balance_dataset(df, label_column='label'):
    majority_class = df[df[label_column] == 0]
    minority_classes = df[df[label_column] != 0]

    oversampled = resample(minority_classes,
                           replace=True,
                           n_samples=len(majority_class),
                           random_state=42)

    return pd.concat([majority_class, oversampled])

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


def train_model():
    # Load data
    df = pd.read_parquet(r'C:\Users\HP\PycharmProjects\VIT\SLP\HAOCD\offenseval_dravidian\malayalam\train-00000-of-00001.parquet')
    df = df[df['label'] != 5]  # Remove 5s
    balanced_df = balance_dataset(df)
    df = balanced_df

    label_mapping = {label: idx for idx, label in enumerate(df['label'].unique())}
    inverse_label_mapping = {idx: label for label, idx in label_mapping.items()}

    with open('label_mapping.pkl', 'wb') as f:
        pickle.dump({'label_mapping': label_mapping,
                     'inverse_label_mapping': inverse_label_mapping}, f)

    df['label_id'] = df['label'].map(label_mapping)

    model_name = "ai4bharat/indic-bert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_mapping)
    )

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].values,
        df['label_id'].values,
        test_size=0.2,
        random_state=42
    )

    train_dataset = TextDataset(train_texts, train_labels, tokenizer)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    num_epochs = 3

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels)

            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels)

                val_loss += outputs.loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f'Epoch {epoch + 1} - Average training loss: {total_loss / len(train_loader):.4f}')
        print(f'Epoch {epoch + 1} - Average validation loss: {avg_val_loss:.4f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            tokenizer.save_pretrained('tokenizer/')


if __name__ == "__main__":
    train_model()