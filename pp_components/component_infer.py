# infer pp components for each of the privacy policy text

# cd pp_components
# python component_infer.py > path/to/log/file

import time
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, BertConfig, AutoModelForSequenceClassification


pp_txts = Path(r"D:\PPAudit\data\pp_txts")
pp_components = Path(r"D:\PPAudit\data\pp_components")
pp_components.mkdir(parents=False, exist_ok=True)

model_name = "pp_components_clf"
tokenizer_ = AutoTokenizer.from_pretrained(model_name)
model_ = AutoModelForSequenceClassification.from_pretrained(model_name)
config_ = BertConfig.from_pretrained(model_name)

id2label = config_.id2label
label2id = config_.label2id


def infer_pp_components(txt_file: Path):
    pred_dict = {
        txt_file.name: {label_: [] for label_ in id2label.values()}
    } 
    # {"pp_url_sha": {
    #   'CUS': [sentence1, sentence2, ...],
    #   'User Choice': [sentence1, sentence2, ...]
    # }} 
    print('[ ] [{}] {}'.format(int(time.time()), txt_file.name))

    with open(txt_file, 'r', encoding='utf-8', errors='?') as rf:
        lines = rf.readlines()
    for line in lines:
        if len(line.split()) < 3:  # ignore this line if it contains less than 3 words 
            continue
        line = line.strip()
        # TODO: to speed up, infer batch instead of singular file
        encoding = tokenizer_(line, return_tensors="pt", padding="max_length", truncation=True, max_length=256)
        encoding = {k: v.to(model_.device) for k, v in encoding.items()}
        outputs = model_(**encoding)
        logits = outputs.logits
        # print(logits.shape)
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(logits.squeeze().cpu())
        predictions = np.zeros(probs.shape)
        predictions[np.where(probs >= 0.5)] = 1
        for idx, prediction in enumerate(predictions):
            if id2label[idx] == "Other":  # do not record 'Other' component
                continue
            if int(prediction) == 1:
                pred_dict[txt_file.name][id2label[idx]].append(line)

    with open(pp_components/'{}.json'.format(txt_file.name), 'w') as wf:
        wf.write(json.dumps(pred_dict, indent=4))

    print('[+] [{}] {}'.format(int(time.time()), txt_file.name))

start_time = time.time()
li = sorted([_ for _ in pp_txts.glob('*.txt')])
for pp_txt in li:
    if (pp_components/'{}.json'.format(pp_txt.name)).is_file():
        print('[-] {}'.format(pp_txt.name))
    else:
        infer_pp_components(pp_txt)
end_time = time.time()
print("[LOG]: infer cost {} s".format(int(end_time-start_time)))





