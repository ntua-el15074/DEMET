from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import RobertaTokenizer, RobertaForSequenceClassification
# from core.logger.logger import Logger
import numpy as np
import torch
import re

TOKENIZER_PATH = '/Users/peteraugerinos/Coding/NTUA Dev/Thesis Umbrella/DEMET/models/roberta-base'
MODEL_PATH = '/Users/peteraugerinos/Coding/NTUA Dev/Thesis Umbrella/DEMET/models/roberta-base'
MODEL_NAME = 'roberta'
EXPLANATION_PATH = '/Users/peteraugerinos/Coding/NTUA Dev/Thesis Umbrella/DEMET/explanations/'

# logger = Logger(True)

if 'roberta' in MODEL_NAME:
    # logger.print_and_log('Using RoBERTa model for explanations','green')
    MODEL = RobertaForSequenceClassification.from_pretrained(MODEL_PATH, ignore_mismatched_sizes=True)
    TOKENIZER = RobertaTokenizer.from_pretrained(TOKENIZER_PATH)
else:
    # logger.print_and_log('Using BERT model for explanations','green')
    TOKENIZER = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    MODEL = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

class ExplainerConfig:
    def __init__(self, method):
        self.tokenizer = TOKENIZER
        self.explanation_path = EXPLANATION_PATH
        self.method = method
        self.model = MODEL
        self.model_name = MODEL_NAME

    def get_model(self):
        return self.model

    def get_model_name(self):
        return self.model_name

    def get_tokenizer(self):
        return self.tokenizer

    def get_explanation_path(self):
        return self.explanation_path

    def get_method(self):
        return self.method


def lime_tokenizer(text):
    cha_tokens = [
        r'\[CHA REPETITION\]',
        r'\[CHA RETRACING\]',
        r'\[CHA SHORT PAUSE\]',
        r'\[CHA MEDIUM PAUSE\]',
        r'\[CHA LONG PAUSE\]',
        r'\[CHA TRAILING OFF\]',
        r'\[CHA PHONOLOGICAL FRAGMENT\]',
        r'\[CHA INTERPOSED WORD\]',
        r'\[CHA FILLER\]',
        r'\[CHA NON COMPLETION OF WORD\]',
        r'\[CHA BELCHES\]',
        r'\[CHA HISSES\]',
        r'\[CHA GRUNTS\]',
        r'\[CHA WHINES\]',
        r'\[CHA COUGHS\]',
        r'\[CHA HUMS\]',
        r'\[CHA ROARS\]',
        r'\[CHA WHISTLES\]',
        r'\[CHA CRIES\]',
        r'\[CHA LAUGHS\]',
        r'\[CHA SNEEZES\]',
        r'\[CHA WHIMPERS\]',
        r'\[CHA GASPS\]',
        r'\[CHA MOANS\]',
        r'\[CHA SIGHS\]',
        r'\[CHA YAWNS\]',
        r'\[CHA GROANS\]',
        r'\[CHA MUMBLES\]',
        r'\[CHA SINGS\]',
        r'\[CHA YELLS\]',
        r'\[CHA GROWLS\]',
        r'\[CHA PANTS\]',
        r'\[CHA SQUEALS\]',
        r'\[CHA VOCALIZES\]',
        r'\[CHA TRAILING OFF QUESTION\]',
        r'\[CHA QUESTION WITH EXCLAMATION\]',
        r'\[CHA INTERRUPTION\]',
        r'\[CHA INTERRUPTION OF QUESTION\]',
        r'\[CHA SELFINTERRUPTION\]',
        r'\[CHA SELFINTERRUPTED QUESTION\]'
    ]
    pattern = '|'.join(cha_tokens) + r'|' + r'\w+'
    return re.findall(pattern, text)


def get_prediction_lime(texts):
    inputs = TOKENIZER(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")

    with torch.no_grad():
        outputs = MODEL(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=-1).numpy()
    return probs

def get_lime_values(exp, label):
    feature_values = exp.as_list(label)
    features, values = zip(*feature_values)
    return features, np.array(values)

def limer(texts, model, tokenizer):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    return probs.detach().numpy()

