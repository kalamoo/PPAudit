# To analyze the completeness of privacy policy text
# fine-tune a multi-label classifier based on PrivBERT

import csv
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModel, BertConfig, PreTrainedModel
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import EvalPrediction
import evaluate
from datasets import Dataset, DatasetDict

from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split

OPP_115_CSV = r"E:\VRppFiles\PP-dataset\OPP-115\annotations"
OPP_115_HTML = r"E:\VRppFiles\PP-dataset\OPP-115\sanitized_policies"

train_ratio = 0.6
validate_ratio = 0.2
test_ratio = 0.2

categories = [
    "First Party Collection/Use",
    "Third Party Sharing/Collection",
    "User Choice/Control",
    "Data Security",
    "International and Specific Audiences",
    "User Access, Edit and Deletion",
    "Policy Change",
    "Data Retention",
    "Do Not Track"
    "Other"
]
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
    #   "First Party Collection/Use" : T/F
    #   "Third Party Sharing/Collection": T/F
    #   .....
    # }
    sample_dict = {"text": text}
    for category in categories:
        if category in labels:
            sample_dict[category] = True
        else:
            sample_dict[category] = False
    return sample_dict


def map_to_multilabels(labels):
    # one-hot label
    # ["Third Party Sharing/Collection", "User Choice/Control"] -> [0, 1, 1, 0, 0...]
    multilabel = [0 for _ in range(len(categories))]
    for label in labels:
        label_idx = categories.index(label)
        multilabel[label_idx] = 1
    return multilabel


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

# ======================================================================================
# Dataset : Full
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

# Dataset : Partial
small_train_dataset = Dataset.from_pandas(dataset_df[:60])
small_validate_dataset = Dataset.from_pandas(dataset_df[60:80])
small_test_dataset = Dataset.from_pandas(dataset_df[80:100])
small_dataset_dict = DatasetDict({
    "train": small_train_dataset,
    "validate": small_validate_dataset,
    "test": small_test_dataset
})

small_encoded_dataset = small_dataset_dict.map(preprocess_data_privbert, batched=True, remove_columns=small_dataset_dict['train'].column_names)
small_encoded_dataset.set_format("torch")

# ======================================================================================


# =====================================================================================


def model_init():
    return AutoModelForSequenceClassification.from_pretrained(
        # "bert-base-uncased",
        "mukund/privbert",  # 换成privBert（ACL‘21） ref https://huggingface.co/mukund/privbert
        problem_type="multi_label_classification",
        num_labels=len(categories),
        id2label=id2label,
        label2id=label2id,
        hidden_dropout_prob=0.15,  # 在训练时使用dropout rate ref https://huggingface.co/xlm-roberta-large/discussions/5
        attention_probs_dropout_prob=0.15,
        # classifier_dropout=0.15  # 根据这个源码解读，设定 classifier_dropout 和 设定 hidden_dropout_prob 效果是一样的 ref https://zhuanlan.zhihu.com/p/441133849
    )


def metrics_chosen(labels_true, labels_pred):
    # 如果需要更多的metrics，可以在这里添加，最根本的还是从 classification_report 中每个类的prec/recall/fl 来计算
    # 记录下每个pp part category各自的precision, recall 和 f1, 
    # 并且选择ave micro f1做为save best model的标准
    # ref https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
    # ref https://towardsdatascience.com/evaluating-multi-label-classifiers-a31be83da6ea
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


# ========================================开始fine tune privbert=============================================


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

    to_train = encoded_dataset["train"] if dataset == "full" else small_encoded_dataset["train"]
    to_eval = encoded_dataset["validate"] if dataset == "full" else small_encoded_dataset["validate"]

    model = model_init()  # 获得priv-bert的bert 模型（即并没有最后的分类的一层）
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
    predictions = trainer.predict(encoded_dataset["test"] if dataset == "full" else small_encoded_dataset["test"])
    print(predictions.metrics)

    trainer.save_model('./ppparts-b')


start_time = time.time()
run_trainer(dataset="full", train_batch_size=16)  # 耗时 5min 左右
end_time = time.time()
print("[LOG]: run fine tune priv-bert trainer cost {} s".format(end_time-start_time))
# [LOG]: run fine tune priv-bert trainer cost 318.3964557647705 s

# ================== 记录 过程中的 test 结果 ========================


# genera setting
# {
# 'test_loss': 0.14657273888587952,
# 'test_macro avg - prec': 0.8869154198985851, 'test_macro avg - recall': 0.7074601052179648, 'test_macro avg - f1': 0.7789654374503822,
# 'test_micro avg - prec': 0.9, 'test_micro avg - recall': 0.7768421052631579, 'test_micro avg - f1': 0.8338983050847457,
# 'test_weighted avg - prec': 0.8970693913405438, 'test_weighted avg - recall': 0.7768421052631579, 'test_weighted avg - f1': 0.8287919772209515,
# 'test_samples avg - prec': 0.921119592875318, 'test_samples avg - recall': 0.8454198473282443, 'test_samples avg - f1': 0.8580225372591783,
# 'test_runtime': 4.847, 'test_samples_per_second': 135.135, 'test_steps_per_second': 16.918
# }

# with dropout rate
# {
# 'test_loss': 0.149748295545578,
# 'test_macro avg - prec': 0.9138701982740964, 'test_macro avg - recall': 0.7026219574472943, 'test_macro avg - f1': 0.7833028255568508,
# 'test_micro avg - prec': 0.9008567931456548, 'test_micro avg - recall': 0.7747368421052632, 'test_micro avg - f1': 0.8330503678551217,
# 'test_weighted avg - prec': 0.899778774783751, 'test_weighted avg - recall': 0.7747368421052632, 'test_weighted avg - f1': 0.8269398242612052,
# 'test_samples avg - prec': 0.9236641221374046, 'test_samples avg - recall': 0.8412213740458016, 'test_samples avg - f1': 0.8584151217739004,
# 'test_runtime': 5.0883, 'test_samples_per_second': 128.727, 'test_steps_per_second': 16.115
# }



# =======================================================================================================
# 用训练好的ppparts来预测

tokenizer_zyx = AutoTokenizer.from_pretrained("ppparts-b")
model_zyx = AutoModelForSequenceClassification.from_pretrained("ppparts-b")


def test_pp_part():
    pred_batch = []
    labels_batch = []
    for sample in test_dataset:
        text = sample["text"]
        labels = [int(label) for label in list(sample.values())[1:]]
        labels_batch.append(labels)

        encoding = tokenizer_zyx(text, return_tensors="pt", padding="max_length", truncation=True, max_length=256)
        encoding = {k: v.to(model_zyx.device) for k, v in encoding.items()}

        outputs = model_zyx(**encoding)
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

# ppparts-a 存入json文件后可作图或做表
# {
    # '0': {'precision': 0.8954248366013072, 'recall': 0.8616352201257862, 'f1-score': 0.8782051282051282, 'support': 318},
    # '1': {'precision': 0.8990825688073395, 'recall': 0.8376068376068376, 'f1-score': 0.8672566371681416, 'support': 234},
    # '2': {'precision': 0.8260869565217391, 'recall': 0.5801526717557252, 'f1-score': 0.6816143497757847, 'support': 131},
    # '3': {'precision': 0.8153846153846154, 'recall': 0.654320987654321, 'f1-score': 0.726027397260274, 'support': 81},
    # '4': {'precision': 0.9393939393939394, 'recall': 0.8985507246376812, 'f1-score': 0.9185185185185185, 'support': 69},
    # '5': {'precision': 0.8421052631578947, 'recall': 0.6956521739130435, 'f1-score': 0.761904761904762, 'support': 46},
    # '6': {'precision': 0.6774193548387096, 'recall': 0.6, 'f1-score': 0.6363636363636364, 'support': 35},
    # '7': {'precision': 0.8888888888888888, 'recall': 0.47058823529411764, 'f1-score': 0.6153846153846153, 'support': 34},
    # '8': {'precision': 1.0, 'recall': 0.8, 'f1-score': 0.888888888888889, 'support': 5},
    # '9': {'precision': 0.8943661971830986, 'recall': 0.7134831460674157, 'f1-score': 0.79375, 'support': 356},
    # 'micro avg': {'precision': 0.8805704099821747, 'recall': 0.7547746371275783, 'f1-score': 0.8128342245989305, 'support': 1309},
    # 'macro avg': {'precision': 0.8678152620777532, 'recall': 0.7111989997054928, 'f1-score': 0.776791393346975, 'support': 1309},
    # 'weighted avg': {'precision': 0.8787435095896409, 'recall': 0.7547746371275783, 'f1-score': 0.8089745059823769, 'support': 1309},
    # 'samples avg': {'precision': 0.9164470794905577, 'recall': 0.8285954785954787, 'f1-score': 0.8412071020766673, 'support': 1309}
# }


start_time = time.time()
test_pp_part()  # 耗时约 3 min
end_time = time.time()
print("[LOG]: run test of pparts-a cost {} s".format(end_time-start_time))
# [LOG]: run test of pparts-a cost 185.49454975128174 s
