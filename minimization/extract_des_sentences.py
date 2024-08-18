import time
from transformers import pipeline
import string
import nltk
import re, json
from pathlib import Path
nltk.download('punkt')
from nltk import word_tokenize, sent_tokenize

output_dir = Path(r'D:\PPAudit\output')
(output_dir/'des_sentences').mkdir(parents=False, exist_ok=True)
app_info_json = Path(r'D:\PPAudit\crawl_data\app_info.json')

class DesClassifier:
    def __init__(self, clf_path):
        self.classifier = pipeline("text-classification", model=clf_path)

    def infer_des(self, text):
        if len(text) > 512: 
            text = text[:512]
        return self.classifier(text)[0]['label']  # P / N
    
des_clf = DesClassifier(r'D:\PPAudit\minimization\description_clf')

def doc_preprocess(doc):

    filter_sentences = []

    sentences = sent_tokenize(doc.replace('\n', ' '))
    # print(sentences)
    
    for sentence in sentences:
        # ignore email / url
        if re.search(r'http(s)?://[^\s]+', sentence) or re.search('www\.[^\s]+', sentence):
            continue  # url
        if re.search(r'[\w-]+(\.[\w-]+)*@[\w-]+((\.[\w-]+)+)', sentence):
            continue  # email addr

        # ignore question (?) sentence
        if sentence.strip()[-1] == '?':
            continue
        
        # delete special characters and duplicated spaces
        sentence = sentence.lower()
        sentence = re.sub(r'[^a-z0-9 ]+', '', sentence)
        sentence = re.sub(r'(\s+)', ' ', sentence)

        tokens = word_tokenize(sentence)  # tokenize
        sentence = ' '.join(tokens)

        filter_sentences.append(sentence)

    return filter_sentences


def extract_feature_sentences(des):
    # extract those sentences with 'description feature'

    # remove email / url; question sentences / special characters / duplicate spaces
    filter_sentences = doc_preprocess(des)

    feature_sentences = []
    for sentence in filter_sentences:
        if des_clf.infer_des(sentence):
            feature_sentences.append(sentence)
    
    return feature_sentences


def extract_for_all_app():
    with open(app_info_json, 'r') as rf:
        app_info = json.load(rf)

    for app_link_hash, item in app_info.items():
        print('[ ] {}'.format(app_link_hash))
        if (output_dir/'des_sentences'/'{}.txt'.format(app_link_hash)).is_file():
            print('[-] {}'.format(app_link_hash))
        try:
            print('[+] {}'.format(app_link_hash))
            des = item['description_complex']
            feature_sentences = extract_feature_sentences(des)
            
            if len(feature_sentences) > 0:
                with open(output_dir/'des_sentences'/'{}.txt'.format(app_link_hash), 'w') as wf:
                    wf.write('\n'.join(feature_sentences))
        except Exception as e:
            print('[Error {}] {}'.format(app_link_hash, str(e)))


start_time = time.time()
extract_for_all_app()
end_time = time.time()
print("[LOG] extract feature sentences cost {} s.".format(end_time - start_time))

