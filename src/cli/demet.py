import os
import sys
import argparse
import requests
import json
from colorama import Style, init
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from core.logger.logger import Logger

init()

os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

logger = Logger(True)

def predict_audio(file_path):
    try:
        url = 'http://localhost:5001/predict_audio'
        files = {'file': open(file_path, 'rb')}
        response = requests.post(url, files=files)
        return response.json()
    except Exception as e:
        error = f'Error: {e}'
        return {'error': error}

def predict_text(text):
    try:
        url = 'http://localhost:5001/predict_text'
        headers = {'Content-Type': 'application/json'}
        data = json.dumps({'text': text})
        response = requests.post(url, headers=headers, data=data)
        return response.json()
    except Exception as e:
        error = f'Error: {e}'
        return {'error': error}

def ansi_colored_text(text, colors):
    return f"\033[38;2;{colors[0]};{colors[1]};{colors[2]}m{text}\033[0m"

def display_explanation(explanation):
    text = "Transcript: "
    for item in explanation:
        feature = item['feature']
        color = item['color']
        colored_feature = ansi_colored_text(feature, color)
        text += f'{colored_feature + Style.RESET_ALL} '
    print(text + Style.RESET_ALL)
    dementia_color = ansi_colored_text('Orange shades', [255, 127, 80])
    non_dementia_color = ansi_colored_text('Blue shades', [100, 149, 237])
    print(f'{dementia_color} indicate Dementia, {non_dementia_color} indicate Non-Dementia' + Style.RESET_ALL)
    print()

def display_prediction(prediction, probability):
    color = [255, 127, 80] if prediction == 'Dementia' else [100, 149, 237]
    colored_prediction = ansi_colored_text(prediction, color)
    colored_probability = ansi_colored_text(str(round(probability * 100, 2)) + '%', color)
    print(f'Prediction: {colored_prediction} ({colored_probability})')

def main():
    parser = argparse.ArgumentParser(description='CLI tool for interacting with the DEMET API.')
    parser.add_argument('--audio', type=str, help='Path to the audio file')
    parser.add_argument('--text', type=str, help='Text to predict')

    args = parser.parse_args()

    if args.audio and args.text:
        parser.error('Please provide either --audio or --text, not both.')

    if args.audio:
        result = predict_audio(args.audio)
        if 'error' in result:
            logger.print(result['error'],"red")
            return
        if 'explanation' in result:
            display_explanation(result['explanation'])
            display_prediction(result['decision'], result['probability'])
    elif args.text:
        result = predict_text(args.text)
        if 'error' in result:
            logger.print(result['error'],"red")
            return
        if 'explanation' in result:
            display_explanation(result['explanation'])
            display_prediction(result['decision'], result['probability'])
    else:
        parser.error('No input provided. Please provide either --audio or --text.')

if __name__ == '__main__':
    main()

