import pandas as pd
import numpy as np

from datasets import Dataset
from transformers import AutoTokenizer
from peft import LoraConfig, TaskType
from transformers import BertForSequenceClassification
from peft import get_peft_model
import evaluate
from transformers import TrainingArguments, Trainer

# Example: Load your dataset
df = pd.read_hdf('./data/in/title_summary_genres_score.h5')

print("data rows before cleaning scores: ",len(df))

df = df[pd.to_numeric(df['meta_score'], errors='coerce').notnull()]

print("data rows after cleaning scores: ",len(df))

df['meta_score'] = pd.to_numeric(df['meta_score'])


dataset = Dataset.from_pandas(df)

dataset = dataset.train_test_split(test_size=0.2)
dataset = dataset.rename_column("meta_score", 'label')


# Extract unique labels from both train and test sets
unique_labels_train = set(dataset['train']['label'])
unique_labels_test = set(dataset['test']['label'])

# Combine unique labels from both sets to ensure we check all possible labels in the dataset
all_unique_labels = unique_labels_train.union(unique_labels_test)

# Define the expected range of labels for the model
expected_labels = set(range(101))  # Since num_labels=93

# Check for mismatches
mismatched_labels = all_unique_labels - expected_labels
if mismatched_labels:
    print(f"Mismatched labels found in dataset: {mismatched_labels}")
else:
    print("No label mismatch detected. Dataset labels match model expectations.")

# Additionally, check for labels expected by the model but missing in the dataset
missing_labels = expected_labels - all_unique_labels
if missing_labels:
    print(f"Labels expected by the model but missing in the dataset: {missing_labels}")
else:
    print("All model's expected labels are present in the dataset.")




tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


def tokenize_function(examples):
    return tokenizer(examples["meta_summary"], padding="max_length", truncation=True)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


tokenized_datasets = dataset.map(tokenize_function, batched=True)

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS, r=1, lora_alpha=1, lora_dropout=0.1
)

model = BertForSequenceClassification.from_pretrained(
    'bert-base-cased',
    num_labels=101
)

model = get_peft_model(model, lora_config)


metric = evaluate.load("accuracy")

training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch",
                                 num_train_epochs=25,)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
)

trainer.train()