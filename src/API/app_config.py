import os
import sys
import torch
import numpy as np
import joblib
from transformers import RobertaTokenizer, BertTokenizer
from transformers import DistilBertTokenizer
from transformers import RobertaForSequenceClassification, BertForSequenceClassification
from lime.lime_text import LimeTextExplainer
import warnings
from transformers import logging
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from core.logger.logger import Logger
from core.explanations.config import lime_tokenizer, get_lime_values, limer
logging.set_verbosity_warning()
logging.set_verbosity_error()
warnings.filterwarnings('ignore')

os.environ['PYTHONPATH'] = '${PYTHONPATH}:/Users/peteraugerinos/Coding/NTUA Dev/Thesis Umbrella/DEMET'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

MODEL_PATH = "/Users/peteraugerinos/Coding/NTUA Dev/Thesis Umbrella/DEMET/models"

logger = Logger(True)

explainer = LimeTextExplainer(class_names=['Non-Dementia', 'Dementia'], split_expression=lime_tokenizer)

shades_of_orange = [
    (255, 165, 0),
    (255, 140, 0),
    (255, 69, 0),
    (255, 99, 71),
    (255, 127, 80),
    (255, 160, 122),
    (255, 215, 0),
    (255, 228, 181),
    (255, 250, 205),
    (255, 239, 213)
]

shades_of_blue = [
    (0, 0, 255),
    (0, 0, 205),
    (0, 0, 139),
    (30, 144, 255),
    (100, 149, 237),
    (135, 206, 250),
    (70, 130, 180),
    (65, 105, 225),
    (25, 25, 112),
    (0, 0, 128)
]

roberta_tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH + '/roberta-base')
bert_tokenizer = BertTokenizer.from_pretrained(MODEL_PATH + '/bert-base')
distil_tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH + '/distilbert')

roberta_model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH + '/roberta-base')
bert_model = BertForSequenceClassification.from_pretrained(MODEL_PATH + '/bert-base')
distil_model = BertForSequenceClassification.from_pretrained(MODEL_PATH + '/distilbert')

models = [roberta_model, bert_model, distil_model]
tokenizers = [roberta_tokenizer, bert_tokenizer, distil_tokenizer]

trained_models = []
for i in range(7):
    try:
        model = joblib.load(MODEL_PATH + f'/ensembles/{i}.joblib')
        trained_models.append(model)
    except Exception as e:
        continue

def extract_logits(model, tokenizer, text):
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)
        return outputs.logits.numpy()

def get_concatenated_logits(text):
    roberta_logits = extract_logits(roberta_model, roberta_tokenizer, text)
    bert_logits = extract_logits(bert_model, bert_tokenizer, text)
    distil_logits = extract_logits(distil_model, distil_tokenizer, text)
    concatenated_logits = np.concatenate((roberta_logits, bert_logits, distil_logits), axis=1)
    return concatenated_logits

def predict(text, method = "single"):
    logger.print("Predicting", "green")
    if method == "ensemble":
        concatenated_logits = get_concatenated_logits(text)
        all_predictions = np.array([model.predict(concatenated_logits) for model in trained_models])
        rounded_predictions = np.round(all_predictions).astype(int)
        majority_vote_predictions = np.apply_along_axis(lambda x: np.bincount(x).argmax() if len(x) > 0 else 0, axis=0, arr=rounded_predictions)
        prediction = int(majority_vote_predictions[0])
        probability = np.mean(all_predictions)
        if prediction == 1:
            decision = 'Dementia'
        else:
            decision = 'Non-Dementia'
        return decision, probability
    if method == "single":
        roberta_model.eval()
        with torch.no_grad():
            inputs = roberta_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            outputs = roberta_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            logger.print(f"Probabilities: {probs}", "green")
            non_dementia_prob = probs[0][0].item()
            dementia_prob = probs[0][1].item()
            if dementia_prob > non_dementia_prob:
                decision = 'Dementia'
                probability = dementia_prob
            else:
                decision = 'Non-Dementia'
                probability = non_dementia_prob

            return decision, probability


def explain(text):
    logger.print("Explaining", "green")
    tokens = lime_tokenizer(text)
    explanation = explainer.explain_instance(text, lambda x: limer(x, roberta_model, roberta_tokenizer), num_features=len(tokens))
    print(explanation.local_exp)
    dementia_features, dementia_values = get_lime_values(explanation, 1)

    explanation_dict = {feature: value for feature, value in zip(dementia_features, dementia_values)}

    explanation = []
    for token in tokens:
        if token in explanation_dict:
            value = explanation_dict[token]
            if value > 0:
                color = shades_of_orange[min(int(value * (len(shades_of_orange) - 1)), len(shades_of_orange) - 1)]
            elif value < 0:
                color = shades_of_blue[min(int(abs(value) * (len(shades_of_blue) - 1)), len(shades_of_blue) - 1)]
            else:
                color = (255, 255, 255)
            explanation.append({'feature': token, 'color': color})
        else:
            default_color = (255, 255, 255)
            explanation.append({'feature': token, 'color': default_color})

    return explanation
