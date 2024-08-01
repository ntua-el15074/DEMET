import warnings
warnings.filterwarnings("ignore")

from transformers import logging
logging.set_verbosity_error()

from lime.lime_text import LimeTextExplainer
from anchor import anchor_text
from datetime import datetime
from transformers_interpret import SequenceClassificationExplainer
from explanations.config import get_prediction_lime, lime_tokenizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model
from logger.logger import Logger
import os
import spacy
import pandas as pd

df = pd.read_csv('/Users/peteraugerinos/Coding/NTUA Dev/Thesis Umbrella/CHA/shuffled_data.csv')
data = df['text'].tolist()
labels = df['gt'].tolist()
train, test, train_labels, test_labels = train_test_split(data, labels, test_size=.2, random_state=42)
train, val, train_labels, val_labels = train_test_split(train, train_labels, test_size=.1, random_state=42)

vectorizer = CountVectorizer(min_df=1)
vectorizer.fit(train)
train_vectors = vectorizer.transform(train)
test_vectors = vectorizer.transform(test)
val_vectors = vectorizer.transform(val)

c = linear_model.LogisticRegression()
c.fit(train_vectors, train_labels)
preds = c.predict(val_vectors)
def predict_lr(texts):
    return c.predict(vectorizer.transform(texts))

logger = Logger(True)

class Explainer:
    def __init__(self, ExplainerConfig):
        self.method = ExplainerConfig.get_method()
        self.explanation_path = ExplainerConfig.get_explanation_path()
        self.tokenizer = ExplainerConfig.get_tokenizer()
        self.model = ExplainerConfig.get_model()
        self.explanation_path += ExplainerConfig.get_method() + '/'
        self.explanation_path += ExplainerConfig.get_model_name() + '/'

        logger.print_and_log(str('Initializing ' + self.method.upper() + ' explainer ...'), 'green')

        if self.method == 'lime':
            self.explainer = LimeTextExplainer(class_names=['Non-Dementia', 'Dementia'], split_expression=lime_tokenizer)

        elif self.method == 'shap':
            # TODO: Implement SHAP
            pass

        elif self.method == 'anchor':
            nlp = spacy.load('en_core_web_sm')
            self.explainer = anchor_text.AnchorText(nlp, ['Non-Dementia', 'Dementia'], use_unk_distribution = True)

        elif self.method == 'transformer-interpret':
            self.explainer = SequenceClassificationExplainer(ExplainerConfig.get_model(), ExplainerConfig.get_tokenizer())

        self.tokenizer = ExplainerConfig.get_tokenizer()
        self.explanation = None
        self.explanation_name = ExplainerConfig.get_model_name()
        self.explanation_name += '_'
        self.explanation_name += datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
        self.timer_start = datetime.now()
        self.timer_end = None

    def explain(self, text):
        logger.print_and_log(str('Explaining with ' + self.method.upper() + ' ...'), 'green')
        try:
            if self.method == 'lime':
                self.explanation = self.explainer.explain_instance(text, get_prediction_lime, num_features=len(text.split(' ')))
                return self.explanation.as_list()

            elif self.method == 'shap':
                # TODO: Implement SHAP
                pass

            elif self.method == 'anchor':
                self.explanation = self.explainer.explain_instance(text, predict_lr, verbose=False)
                pred = self.explainer.class_names[predict_lr([text])[0]]
                alternative =  self.explainer.class_names[1 - predict_lr([text])[0]]
                temp = ""
                temp += 'Examples where anchor applies and model predicts %s:\n' % pred
                temp += '\n'.join([x[0] for x in self.explanation.examples(only_same_prediction=True)])
                temp += '\n\nExamples where anchor applies and model predicts %s:\n' % alternative
                temp += '\n'.join([x[0] for x in self.explanation.examples(only_different_prediction=True)])
                self.explanation = self.explainer.show_in_notebook(text, predict_lr)
                self.explanation = temp
                return self.explanation

            elif self.method == 'transformer-interpret':
                self.explanation = self.explainer(text, class_name='Dementia')
                return self.explanation

        except Exception as e:
            logger.print_and_log('Error in explaining: ' + str(e), 'red')

    def save(self):
        logger.print_and_log('Saving explanation...', 'green')
        self.timer_end = datetime.now()
        logger.print_and_log('Explanation took: ' + str(self.timer_end - self.timer_start), 'green')
        if not os.path.exists(self.explanation_path):
            os.makedirs(self.explanation_path, exist_ok=True)
        try:
            if self.method == 'lime':
                self.explanation.save_to_file(self.explanation_path + self.explanation_name + '.html')

            elif self.method == 'shap':
                # TODO: Implement SHAP
                pass

            elif self.method == 'anchor':
                with open(self.explanation_path + self.explanation_name + '.txt', 'w') as f:
                    f.write(str(self.explanation))

            elif self.method == 'transformer-interpret':
                self.explainer.visualize(self.explanation_path + self.explanation_name + '.html')

            with open(self.explanation_path + 'times.txt', 'a') as f:
                f.write(str(self.timer_end - self.timer_start) + '\n')

        except Exception as e:
            logger.print_and_log('Error in saving explanation: ' + str(e), 'red')

