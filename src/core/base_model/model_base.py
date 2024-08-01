import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from logger.logger import Logger
from config import ModelBaseConfig
from transformers import BertForSequenceClassification, BertTokenizer, RobertaForSequenceClassification, RobertaTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, RandomSampler, TensorDataset, Subset
from transformers import AdamW
from sklearn.model_selection import KFold
import pandas as pd

logger = Logger()

class ModelBase:
    def __init__(self,ModelBaseConfig):
        self.model_name = ModelBaseConfig.get_model_name()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.extra_tokens = ModelBaseConfig.get_extra_tokens()
        if 'roberta' in self.model_name:
            logger.print("Initializing RoBERTa Model ...", "green")
            self.model = RobertaForSequenceClassification.from_pretrained(self.model_name, num_labels=2).to(self.device)
            self.tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
        else:
            logger.print("Initializing BERT Model ...", "green")
            self.model = BertForSequenceClassification.from_pretrained(self.model_name, num_labels=2).to(self.device)
            self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        if self.extra_tokens:
            self.add_tokens()

    def add_tokens(self):
        try:
            logger.print("Adding Extra Tokens ...", "green")
            self.tokenizer.add_tokens(self.extra_tokens, special_tokens = True)
            self.model.resize_token_embeddings(len(self.tokenizer))
        except Exception as e:
            logger.print("Error: " + str(e), "red")

    def train_epoch(self,dataloader,optimizer):
        self.model.train()
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
        return total_loss / len(dataloader)

    def evaluate(self, dataloader):
        self.model.eval()
        predictions = []
        actual_labels = []
        with torch.no_grad():
            for batch in dataloader:
                input_ids, attention_mask, labels = batch
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_output = torch.argmax(outputs.logits, dim=1).cpu().detach().numpy()
                val_labels = labels.cpu().detach().numpy()
                predictions.extend(val_output)
                actual_labels.extend(val_labels)
        accuracy = accuracy_score(actual_labels, predictions)
        precision = precision_score(actual_labels, predictions)
        recall = recall_score(actual_labels, predictions)
        f1 = f1_score(actual_labels, predictions)
        matrix = confusion_matrix(actual_labels, predictions)
        logger.print("Evaluation Results.", "green")
        logger.print("Accuracy: " + str(accuracy), "green")
        logger.print("Precision: " + str(precision), "green")
        logger.print("Recall: " + str(recall), "green")
        logger.print("F1 Score: " + str(f1), "green")
        return accuracy, precision, recall, f1, matrix

    def save_model(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.print("Model Saved Successfully.", "green")


def cross_validation(model_name, data, labels, epochs, batch_size, learning_rate, n_splits):
    kfold = KFold(n_splits=n_splits, shuffle=True)
    C = ModelBaseConfig(model_name)
    M = ModelBase(C)
    encodings = M.tokenizer(data, truncation=True, padding=True, return_tensors="pt")
    dataset = TensorDataset(encodings['input_ids'], encodings['attention_mask'], torch.tensor(labels))

    scores_matrix = np.zeros((n_splits, epochs))

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        path = ""
        scores_path = CORE_PATH + str("scores")
        if 'roberta' in model_name:
          path = scores_path + '/roberta'
        elif 'distil' in model_name:
          path = scores_path + '/distil'
        else:
          path = scores_path + '/base'
        if not os.path.exists(path):
          os.makedirs(path)
        path = f"{path}/scores_{fold}.csv"
        with open(path, 'a') as f:
          f.write("accuracy,precision,recall,f1\n")
          config = ModelBaseConfig(model_name)
          model = ModelBase(config)
          logger.print(f"Training Model - Fold {fold+1}/{n_splits} ...","green")

          train_subset = Subset(dataset, train_idx)
          val_subset = Subset(dataset, val_idx)

          train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
          val_loader = DataLoader(val_subset, batch_size=batch_size)

          optimizer = AdamW(model.model.parameters(), lr=learning_rate, weight_decay=0.01)

          for epoch in range(epochs):
              logger.print(f"Epoch {epoch + 1}/{epochs}","green")
              train_loss = model.train_epoch(train_loader, optimizer)
              logger.print(f"Train loss: {train_loss}","green")

              accuracy, precision, recall, f1, matrix = model.evaluate(val_loader)
              f.write(f"{accuracy},{precision},{recall},{f1}\n")
              plot_confusion(matrix, ['Non-Dementia', 'Dementia'], model_name, fold, epoch)

              logger.print(f"Validation accuracy: {accuracy}","green")
              scores_matrix[fold, epoch] = accuracy

          #model.save_model(f"{CORE_PATH}models/{model_name}_{fold}")
    return scores_matrix



def plot_confusion(conf_matrix, labels, model_name, fold, epoch):
    title = f"{model_name} - Fold {fold} - Epoch {epoch}"
    if not os.path.exists(CORE_PATH + str("plots")):
      os.makedirs(CORE_PATH + str("plots"))
    path = f"{CORE_PATH}plots/{model_name}_fold_{fold}_epoch_{epoch}.png"
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.savefig(path)

def save_scores(scores_matrix, model_name):
    scores_df = pd.DataFrame(scores_matrix)
    scores_df.to_csv(f"scores/{model_name}_scores.csv")
