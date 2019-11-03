import sys
import glob
sys.path.append('./classification')
sys.path.append('..')

from app import app
from classification.detect_from_video import test_full_image_network
from classification.detect_from_image import detect_from_image
from classification.network import models
import torch
# from app.download_yt import download_video
from compression_detection import compression_detection
import os
from flask import Flask, flash, request, redirect, url_for, render_template

from .scraper import *
from .text_detection import *


UPLOAD_FOLDER = './classification/data_dir/uploads'
ALLOWED_EXTENSIONS = set(['html', '/'])
MODEL_PATH = './classification/weights/full/xception/full_c23.p'
OUTPUT_PATH = './classification/data_dir/results'

# app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

cuda = False

base_weights_path = 'classification/weights/face_detection/xception'
model_full_path = f'{base_weights_path}/all_raw.p'
model_77_path = f'{base_weights_path}/all_c23.p'
model_60_path = f'{base_weights_path}/all_c40.p'

model_full = torch.load(model_full_path, map_location=lambda storage, loc: storage)
model_77 = torch.load(model_77_path, map_location=lambda storage, loc: storage)
model_60 = torch.load(model_60_path, map_location=lambda storage, loc: storage)

gpt = LM()

# dbio = DatabaseIO()

@app.route('/')
def index():
    return render_template('index.html')


# POST IMAGE
@app.route('/snooop', methods=['GET', 'POST'])
def check_if_fake():
    try:
        if request.method =='POST':
            url = request.form['get_link']
            text_preds = []
            image_preds = []
            fakes = []

            scraped = get_elements(url)
            if scraped[0]:
                raw_text = ''.join(scraped[0]).encode('ascii', 'replace').decode()
                result_percentage = get_generated_analysis(raw_text, gpt)
                if result_percentage >= 0.3:
                    text_preds.append('very low likelihood')
                if result_percentage >= 0.1:
                    text_preds.append('low likelihood')
                elif result_percentage >= 0.09:
                    text_preds.append('reasonable chance')
                else:
                    text_preds.append('high likelihood')
                text_preds.append(result_percentage)

            if scraped[1]:
                print('found images, running detection')
                for image in scraped[1]:
                    image_preds.append((detect_from_image(image, model_full, cuda=cuda), image))
                print(image_preds)
                fakes = [i for i in image_preds if i[0] == 0]


            if scraped[2]:
                print('found videos, running detection')
                predicted_class = compression_detection.classify_video(scraped[2])

                if predicted_class == '0.6':
                    fake_prediction = test_full_image_network(scraped[2], model=model_60, output_path=OUTPUT_PATH,
                                            start_frame=0, end_frame=None, cuda=cuda)
                elif predicted_class == '0.77':
                    fake_prediction = test_full_image_network(scraped[2], model=model_77, output_path=OUTPUT_PATH,
                                            start_frame=0, end_frame=None, cuda=cuda)
                elif predicted_class == 'original':
                    fake_prediction = test_full_image_network(scraped[2], model=model_full, output_path=OUTPUT_PATH,
                                            start_frame=0, end_frame=None, cuda=cuda)
                else:
                    fake_prediction = None

                print(f'fake_prediction: {fake_prediction}')
                # if fake_prediction is not None:
                #     history = history.append({'hash': hash, 'link': youtube_url, 'filename': upload_fname,
                #                               'fake': fake_prediction}, ignore_index=True)
                #     dbio.write_history(history)

                # os.remove(filepath)

                if fake_prediction == 1:
                    a = False
                    print(a)
                    # return render_template('fake.hmtl')
                elif fake_prediction == 0:
                    a = True
                    print(a)
                    # return render_template('real.html')
                else:
                    flash('ERROR! Something went wrong. Please try again.')
                    return redirect(url_for('index'))
                    # return render_template('error.html')

        else:
            flash('ERROR! Something went wrong. Please try again.')
            return redirect(url_for('index'))

        return render_template('results.html', posOfAI = text_preds, num_images=len(image_preds), fakes=fakes)

    except:
        flash('ERROR! Something went wrong. Please try again.')
        return redirect(url_for('index'))
