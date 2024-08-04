
import time
import numpy as np
import pandas as pd
import random
random.seed(2023)
from datasets import Dataset, DatasetDict
# NOTE: package may be incompatible
# datasets 1.11.0 requires huggingface-hub<0.1.0, but you have huggingface-hub 0.4.0 which is incompatible.
from transformers import AutoTokenizer, DataCollatorForTokenClassification, Trainer, AutoModelForTokenClassification, TrainingArguments
from seqeval.metrics import accuracy_score, precision_score, recall_score, f1_score
# pip install seqval
from synthetic_cus import synthetic_cus

# pip install 
#  + seqeval
label_names = ['O', 'B', 'I']

model_checkpoint = "mukund/privbert"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

cus_words, cus_labels, ncus_words = synthetic_cus()
neg2pos_ratio = 2  # neg / pos = 2 : 1 
ncus_words = ncus_words[:int(neg2pos_ratio * len(cus_words))]


train_ratio = 0.7
validate_ratio = 0.2
test = 0.1

data = [{
    "ner_tags": labels,
    "words": words
} for (labels, words) in zip(cus_labels, cus_words)] + [{
    "ner_tags": [0 for _ in words],  # all 'O' -> label=0
    "words": words
} for words in ncus_words]
random.shuffle(data)
data_df = pd.DataFrame(data)

train_dataset = Dataset.from_pandas(data_df[:int(train_ratio*len(data_df))])
validate_dataset = Dataset.from_pandas(data_df[int(train_ratio*len(data_df)):int((train_ratio+validate_ratio)*len(data_df))])
test_dataset = Dataset.from_pandas(data_df[int((train_ratio+validate_ratio)*len(data_df)):])

raw_datasets = DatasetDict({
    "train": train_dataset,
    "validation": validate_dataset,
    "test": test_dataset
})


def align_labels_with_tokens(labels, word_ids):
    # Input：
    #   labels: label of words (len=N)
    #   word_ids: word idx of each token (len=M)
    # Output：
    #   new_labels: label of tokens (len=M)
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # chage label of B (here is the remaining 'tokens' of a B 'word') to I
            if label == 1:
                label = 2
            new_labels.append(label)
    
    return new_labels


def tokenize_and_align_labels(examples):
    # 'word' in each sentence may be tokenized into multiple 'tokens'
    tokenized_inputs = tokenizer(
        examples["words"], truncation=True, is_split_into_words=True
    )
    all_labels = examples["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))
    
    # add 'labels' column to the tokenized_inputs
    # now it has the following 4 column
    # input_idx / token_type_ids / attention_mask / labels
    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs


tokenized_datasets = raw_datasets.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=raw_datasets["test"].column_names,
)


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    
    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    return {
        "precision": precision_score(y_true=true_labels, y_pred=true_predictions),
        "recall": recall_score(y_true=true_labels, y_pred=true_predictions),
        "f1": f1_score(y_true=true_labels, y_pred=true_predictions),
        "accuracy": accuracy_score(y_true=true_labels, y_pred=true_predictions),
    }
    

id2label = {i: label for i, label in enumerate(label_names)}
label2id = {v: k for k, v in id2label.items()}
model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    id2label=id2label,
    label2id=label2id,
)

args = TrainingArguments(
    "dataobj_ner_checkpoint",
    evaluation_strategy="steps",
    learning_rate=2e-5,
    num_train_epochs=20,
    weight_decay=0.01,
    metric_for_best_model="f1",
    save_total_limit=3,
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)

start_time = time.time()

trainer.train()
predictions = trainer.predict(tokenized_datasets["test"])
print(predictions.metrics)
trainer.save_model('./dataobj_ner')

end_time = time.time()
print("[LOG]: run dataobj ner trainer cost {} s".format(end_time - start_time))