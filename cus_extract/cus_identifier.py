import time
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict
from sklearn.metrics import classification_report
from synthetic_cus import synthetic_cus
import random
random.seed(2023)


cus_labels = [
    "N",  # 0
    "P"   # 1
]

id2label = {idx: label for idx, label in enumerate(cus_labels)}
label2id = {label: idx for idx, label in enumerate(cus_labels)}

train_ratio = 0.7
validate_ratio = 0.2
test = 0.1

# CUS and non-CUS sentences
cus_words, _, ncus_words = synthetic_cus()

cus_train = cus_words[:int(train_ratio*len(cus_words))]
cus_validate = cus_words[int(train_ratio*len(cus_words)):int((train_ratio+validate_ratio)*len(cus_words))]
cus_test = cus_words[int((train_ratio+validate_ratio)*len(cus_words)):]

ncus_train = ncus_words[:int(train_ratio*len(ncus_words))]
ncus_validate = ncus_words[int(train_ratio*len(ncus_words)):int((train_ratio+validate_ratio)*len(ncus_words))]
ncus_test = ncus_words[int((train_ratio+validate_ratio)*len(ncus_words)):]

full_train = [{
    "text": ' '.join(tokens),
    "label": 1
} for tokens in cus_train] + [{
    "text": ' '.join(tokens),
    "label": 0
} for tokens in ncus_train]
random.shuffle(full_train)

full_validate = [{
    "text": ' '.join(tokens),
    "label": 1
} for tokens in cus_validate] + [{
    "text": ' '.join(tokens),
    "label": 0
} for tokens in ncus_validate]

full_test = [{
    "text": ' '.join(tokens),
    "label": 1
} for tokens in cus_test] + [{
    "text": ' '.join(tokens),
    "label": 0
} for tokens in ncus_test]

raw_datasets = DatasetDict({
    "train": Dataset.from_pandas(pd.DataFrame(full_train)),
    "validation": Dataset.from_pandas(pd.DataFrame(full_validate)),
    "test": Dataset.from_pandas(pd.DataFrame(full_test))
})


privbert_tokenizer = AutoTokenizer.from_pretrained("mukund/privbert")

def preprocess_data_privbert(examples):
    encoding = privbert_tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)
    return encoding

encoded_dataset = raw_datasets.map(preprocess_data_privbert, batched=True)
encoded_dataset.set_format("torch")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return classification_report(labels, predictions, output_dict=True)


def run_trainer(train_batch_size=16, l_rate=2.5e-5, num_epochs=3):
    
    training_args = TrainingArguments(
        f"privbert-finetuned-for-cus-identification",
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
        "mukund/privbert",  # ref https://huggingface.co/mukund/privbert
        num_labels=len(cus_labels),
        id2label=id2label,
        label2id=label2id,
    )
    
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
    predictions = trainer.predict(encoded_dataset["test"])
    print(predictions.metrics)
    trainer.save_model('./cus_clf')


start_time = time.time()
run_trainer() 
end_time = time.time()
print("[LOG]: run cus clf trainer cost {} s".format(end_time - start_time))


