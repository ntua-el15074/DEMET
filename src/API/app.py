#!/usr/bin/env python3
import os
import sys
from flask import Flask, request, jsonify
from flask_cors import CORS
from app_config import predict, explain
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from core.processing.config import ProcessingConfig
from core.processing.processor import Processor
from core.logger.logger import Logger

os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

app = Flask(__name__)
CORS(app)

logger = Logger(True)
logger.print("Welcome to DEMET v0.1", "green")
logger.print("Initializing BERT Models", "green")

@app.route('/predict_audio', methods=['POST'])
def predict_audio_route():
    file = request.files['file']
    if file:
        processor_conf = ProcessingConfig()
        processor = Processor(processor_conf)
        text = processor.transribe_audio(file)
        decision, probability = predict(text)
        logger.print("Prediction: " + decision, "green")
        explanation = explain(text)
        return jsonify({'decision': decision, 'probability': probability, 'explanation': explanation})
    else:
        return jsonify({'error': 'No file provided'})

@app.route('/predict_text', methods=['POST'])
def predict_text_route():
    text = request.json['text']
    decision, probability = predict(text)
    explanation = explain(text)
    logger.print("Prediction: " + decision, "green")
    return jsonify({'decision': decision, 'probability': probability, 'explanation': explanation})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)

