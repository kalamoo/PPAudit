# To analyze the completeness of privacy policy text
# fine-tune a multi-label classifier based on PrivBERT

import csv
import os
import re
import time
from pathlib import Path
import numpy as np

# for CPU-only 
# pip install 
# + transformers[torch]
# + pandas
# + scikit-learn
# + chardet   # to solve possible issue: ImportError: cannot import name 'get_full_repo_name' from 'huggingface_hub

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EvalPrediction
from datasets import Dataset, DatasetDict
from sklearn.metrics import classification_report

# TODO: change to cmd and take args

OPP_115_dir = Path(r'D:\BrowserDownload\OPP-115_v1_0\OPP-115')  # path/to/OPP_115 dir
# download here : https://usableprivacy.org/data
OPP_115_CSV = OPP_115_dir / 'annotations'
OPP_115_HTML = OPP_115_dir / 'sanitized_policies'

model_save_name = "pp_components_clf"

train_ratio = 0.6
validate_ratio = 0.2
test_ratio = 0.2

# so far, ignore DNT & Other; & merge CUS
categories_mapping = {
    "First Party Collection/Use": "CUS",
    "Third Party Sharing/Collection": "CUS",
    "User Choice/Control": "User Choice",
    "Data Security": "Data Security",
    "International and Specific Audiences": "Specific Audiences",
    "User Access, Edit and Deletion": "User Rights",
    "Policy Change": "Policy Change",
    "Data Retention": "Data Retention",
    "Do Not Track": "Other",
    "Other": "Other"
}

categories = ["CUS", "User Choice", "Data Security", "Specific Audiences", "User Rights", "Policy Change", "Data Retention", "Other"]

id2label = {idx: label for idx, label in enumerate(categories)}
label2id = {label: idx for idx, label in enumerate(categories)}

# ref https://huggingface.co/mukund/privbert
privbert_tokenizer = AutoTokenizer.from_pretrained("mukund/privbert")


def clean_segment(segment):
    # remove html tag in segment
    segment = re.sub(r"<(/)?strong>|<br>|<(/)?ul>|<(/)?li>", "", segment)
    segment = re.sub(r"(\s){2,}", " ", segment)
    segment = segment.strip()
    return segment


def construct_sample(text, labels):
    # construct the training data for finetuning PrivBERT
    # {
    #   "text": "xxx",
    #   "CUS" : T/F
    #   "User Choice": T/F
    #   .....
    # }
    # print(labels)
    sample_dict = {"text": text}
    for category in categories:
        if category in labels:
            sample_dict[category] = True
        else:
            sample_dict[category] = False
    # print(sample_dict)
    return sample_dict


def prepare_dataset():
    # preprocess OPP-115 data (add label for each entry)
    texts_labels = []
    for filename in os.listdir(OPP_115_CSV):
        appname = os.path.splitext(filename)[0]
        htmlfilepath = str(Path(OPP_115_HTML, "{}.{}".format(appname, "html")))
        segment_labels_pair = dict()  # {segment_idx: segment_text, labels(list)}
        with open(htmlfilepath, 'r') as rf:
            html_content = rf.read()
            segments = html_content.split("|||")
            segments = [clean_segment(segment) for segment in segments]

        csvfilepath = str(Path(OPP_115_CSV, filename))
        with open(csvfilepath, 'r') as rf:
            csv_reader = csv.reader(rf)
            for row_index, row in enumerate(csv_reader):
                _, _, _, _, segment_idx, category, _, _, _ = row
                category = categories_mapping[category]
                if category not in categories:
                    continue
                if segment_idx not in segment_labels_pair.keys():
                    segment_labels_pair[segment_idx] = dict()
                    segment_labels_pair[segment_idx]["segment_text"] = segments[int(segment_idx)]
                    segment_labels_pair[segment_idx]["segment_labels"] = [category, ]
                else:
                    segment_labels_pair[segment_idx]["segment_labels"].append(category)

        for segment_idx, segment_infos in segment_labels_pair.items():
            segment_text = segment_labels_pair[segment_idx]["segment_text"]
            segment_labels = list(set(segment_labels_pair[segment_idx]["segment_labels"]))
            sample = construct_sample(segment_text, segment_labels)
            texts_labels.append(sample)
    return pd.DataFrame(texts_labels)


def preprocess_data_privbert(examples):
    # to finetune privbert, we need to encode text with BERT tokenizer
    
    # take a batch of texts
    text = examples["text"]
    # encode them
    encoding = privbert_tokenizer(text, padding="max_length", truncation=True, max_length=256)
    # add labels
    labels_batch = {k: examples[k] for k in examples.keys() if k in categories}
    # create numpy array of shape (batch_size, num_labels)
    labels_matrix = np.zeros((len(text), len(categories)))
    # fill numpy array
    for idx, label in enumerate(categories):
        labels_matrix[:, idx] = labels_batch[label]
    encoding["labels"] = labels_matrix.tolist()

    return encoding


dataset_df = prepare_dataset()

train_dataset = Dataset.from_pandas(dataset_df[:int(train_ratio*len(dataset_df))])
validate_dataset = Dataset.from_pandas(dataset_df[int(train_ratio*len(dataset_df)):int((train_ratio+validate_ratio)*len(dataset_df))])
test_dataset = Dataset.from_pandas(dataset_df[int((train_ratio+validate_ratio)*len(dataset_df)):])
dataset_dict = DatasetDict({
    "train": train_dataset,
    "validate": validate_dataset,
    "test": test_dataset
})
encoded_dataset = dataset_dict.map(preprocess_data_privbert, batched=True, remove_columns=dataset_dict['train'].column_names)
encoded_dataset.set_format("torch")

# small_train_dataset = Dataset.from_pandas(dataset_df[:60])
# small_validate_dataset = Dataset.from_pandas(dataset_df[60:80])
# small_test_dataset = Dataset.from_pandas(dataset_df[80:100])
# small_dataset_dict = DatasetDict({
#     "train": small_train_dataset,
#     "validate": small_validate_dataset,
#     "test": small_test_dataset
# })

# small_encoded_dataset = small_dataset_dict.map(preprocess_data_privbert, batched=True, remove_columns=small_dataset_dict['train'].column_names)
# small_encoded_dataset.set_format("torch")


def model_init():
    return AutoModelForSequenceClassification.from_pretrained(
        "mukund/privbert",  # ref https://huggingface.co/mukund/privbert
        problem_type="multi_label_classification",
        num_labels=len(categories),
        id2label=id2label,
        label2id=label2id,
        hidden_dropout_prob=0.15,  # dropout rate ref https://huggingface.co/xlm-roberta-large/discussions/5
        attention_probs_dropout_prob=0.15,
        # classifier_dropout=0.15 # same as setting hidden_dropout_prob
    )


def metrics_chosen(labels_true, labels_pred):
    # choose ave micro f1 as metrics for save best model
    # ref https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
    metrics_full = classification_report(labels_true, labels_pred, output_dict=True)

    metrics_chosen = dict()

    metrics_chosen["macro avg - prec"] = metrics_full["macro avg"]["precision"]
    metrics_chosen["macro avg - recall"] = metrics_full["macro avg"]["recall"]
    metrics_chosen["macro avg - f1"] = metrics_full["macro avg"]["f1-score"]

    metrics_chosen["micro avg - prec"] = metrics_full["micro avg"]["precision"]
    metrics_chosen["micro avg - recall"] = metrics_full["micro avg"]["recall"]
    metrics_chosen["micro avg - f1"] = metrics_full["micro avg"]["f1-score"]

    metrics_chosen["weighted avg - prec"] = metrics_full["weighted avg"]["precision"]
    metrics_chosen["weighted avg - recall"] = metrics_full["weighted avg"]["recall"]
    metrics_chosen["weighted avg - f1"] = metrics_full["weighted avg"]["f1-score"]

    metrics_chosen["samples avg - prec"] = metrics_full["samples avg"]["precision"]
    metrics_chosen["samples avg - recall"] = metrics_full["samples avg"]["recall"]
    metrics_chosen["samples avg - f1"] = metrics_full["samples avg"]["f1-score"]

    """
        f1_macro_average = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
        roc_auc = roc_auc_score(y_true, y_pred, average='macro')
        accuracy = accuracy_score(y_true, y_pred)

        # return as dictionary
        metrics = {'f1': f1_macro_average,
                   'roc_auc': roc_auc,
                   'accuracy': accuracy}
    """

    return metrics_chosen


def multi_label_metrics(logits, labels, threshold=0.5):
    # first, apply sigmoid on logits which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(logits))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels

    return metrics_chosen(y_true, y_pred)


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    result = multi_label_metrics(
        logits=preds,
        labels=p.label_ids)
    return result


# Train


def run_trainer(dataset="full", train_batch_size=16, l_rate=2.5e-5, num_epochs=4):
    training_args = TrainingArguments(
        f"privbert-finetuned-for-pp-parts",
        learning_rate=l_rate,
        evaluation_strategy="epoch",
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=16,
        num_train_epochs=num_epochs,
        metric_for_best_model="micro avg - f1",
        save_strategy="no",
        run_name="{}-{}-{}-{}".format(dataset, train_batch_size, l_rate, num_epochs)
        # push_to_hub=True,
    )

    # to_train = encoded_dataset["train"] if dataset == "full" else small_encoded_dataset["train"]
    # to_eval = encoded_dataset["validate"] if dataset == "full" else small_encoded_dataset["validate"]
    
    to_train = encoded_dataset["train"]
    to_eval = encoded_dataset["validate"] 

    model = model_init() 
    # pretty_print(model)
    trainer = Trainer(
        model,
        args=training_args,
        train_dataset=to_train,
        eval_dataset=to_eval,
        tokenizer=privbert_tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

    # test on testset
    # predictions = trainer.predict(encoded_dataset["test"] if dataset == "full" else small_encoded_dataset["test"])
    predictions = trainer.predict(encoded_dataset["test"])
    print(predictions.metrics)

    trainer.save_model(model_save_name)


start_time = time.time()
run_trainer(dataset="full", train_batch_size=16)
end_time = time.time()
print("[LOG]: run fine tune priv-bert trainer cost {} s".format(int(end_time-start_time)))

# Test
tokenizer_ = AutoTokenizer.from_pretrained(model_save_name)
model_ = AutoModelForSequenceClassification.from_pretrained(model_save_name)

def test_clf():
    pred_batch = []
    labels_batch = []
    for sample in test_dataset:
        text = sample["text"]
        labels = [int(label) for label in list(sample.values())[1:]]
        labels_batch.append(labels)

        encoding = tokenizer_(text, return_tensors="pt", padding="max_length", truncation=True, max_length=256)
        encoding = {k: v.to(model_.device) for k, v in encoding.items()}

        outputs = model_(**encoding)
        logits = outputs.logits
        # print(logits.shape)

        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(logits.squeeze().cpu())
        predictions = np.zeros(probs.shape)
        predictions[np.where(probs >= 0.5)] = 1

        pred_batch.append(predictions)

    metrics = metrics_chosen(labels_batch, pred_batch)
    metrics_full = classification_report(labels_batch, pred_batch, output_dict=True)
    print(metrics_full)


start_time = time.time()
test_clf()
end_time = time.time()
print("[LOG]: run test cost {} s".format(int(end_time-start_time)))

