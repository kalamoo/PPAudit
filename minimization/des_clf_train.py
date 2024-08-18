import time
from pathlib import Path
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict
from sklearn.metrics import classification_report
import random

random.seed(2022)

des_sentences_dir = Path(r'D:\PPAudit\other_data\feature-sentences-dataset')

des_labels = [
    "N",  # 0  non-feature
    "P"  # 1  feature
]

id2label = {idx: label for idx, label in enumerate(des_labels)}
label2id = {label: idx for idx, label in enumerate(des_labels)}

# non-des or description featured sentences
def read_features_txt(path):
    with open(path, 'r') as rf:
        return rf.readlines()
des_sentences = read_features_txt(des_sentences_dir / 'features.txt')
ndes_sentences = read_features_txt(des_sentences_dir / 'non-features.txt')

train_ratio = 0.7
validate_ratio = 0.2
test = 0.1

des_train = des_sentences[:int(train_ratio * len(des_sentences))]
des_validate = des_sentences[int(train_ratio * len(des_sentences)):int((train_ratio + validate_ratio) * len(des_sentences))]
des_test = des_sentences[int((train_ratio + validate_ratio) * len(des_sentences)):]

ndes_train = ndes_sentences[:int(train_ratio * len(ndes_sentences))]
ndes_validate = ndes_sentences[int(train_ratio * len(ndes_sentences)):int(
    (train_ratio + validate_ratio) * len(ndes_sentences))]
ndes_test = ndes_sentences[int((train_ratio + validate_ratio) * len(ndes_sentences)):]

full_train = [{
    "text": sentence,
    "label": 1
} for sentence in des_train] + [{
    "text": sentence,
    "label": 0
} for sentence in ndes_train]
random.shuffle(full_train)

full_validate = [{
    "text": sentence,
    "label": 1
} for sentence in des_validate] + [{
    "text": sentence,
    "label": 0
} for sentence in ndes_validate]

full_test = [{
    "text": sentence,
    "label": 1
} for sentence in des_test] + [{
    "text": sentence,
    "label": 0
} for sentence in ndes_test]

raw_datasets = DatasetDict({
    "train": Dataset.from_pandas(pd.DataFrame(full_train)),
    "validation": Dataset.from_pandas(pd.DataFrame(full_validate)),
    "test": Dataset.from_pandas(pd.DataFrame(full_test))
})

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


def preprocess_data_privbert(examples):
    # tokenizer encode
    encoding = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)
    return encoding


encoded_dataset = raw_datasets.map(preprocess_data_privbert, batched=True)
encoded_dataset.set_format("torch")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return classification_report(labels, predictions, output_dict=True)


def run_trainer(train_batch_size=16, l_rate=2.5e-5, num_epochs=3):
    
    training_args = TrainingArguments(
        f"bert-finetuned-for-des-identification",
        learning_rate=l_rate,
        evaluation_strategy="epoch",
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=16,
        num_train_epochs=num_epochs,
        metric_for_best_model="accuracy",
        save_strategy="no",
    )
    
    to_train = encoded_dataset["train"]
    to_eval = encoded_dataset["validation"]
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-cased",
        num_labels=len(des_labels),
        id2label=id2label,
        label2id=label2id
    )
    
    trainer = Trainer(
        model,
        args=training_args,
        train_dataset=to_train,
        eval_dataset=to_eval,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    
    trainer.train()
    
    # test on testset
    predictions = trainer.predict(encoded_dataset["test"])
    print(predictions.metrics)
    trainer.save_model('./description_clf')



start_time = time.time()
run_trainer() 
end_time = time.time()
print("[LOG]: run fine tune bert trainer cost {} s".format(end_time - start_time))


