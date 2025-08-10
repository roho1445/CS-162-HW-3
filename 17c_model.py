from datasets import load_dataset

# Load the mafand dataset
mafand = load_dataset('masakhane/mafand', 'en-amh')

# Print the available splits
print("Available splits:", mafand.keys())

# Create train/val split from validation data
train_val = mafand['validation'].train_test_split(test_size=0.2, seed=42)
train_data = train_val['train']
val_data = train_val['test']
test_data = mafand['test']

# Print the size of each split
print(f"\nTraining examples: {len(train_data)}")
print(f"Validation examples: {len(val_data)}")
print(f"Test examples: {len(test_data)}")

# Print a sample from the training set
print("\nSample training example:")
print(train_data[0])

from transformers import AutoTokenizer

checkpoint = "google-t5/t5-base"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

source_lang = "amh"
target_lang = "en"
prefix = "translate Amharic to English: "


def preprocess_function(examples):
    inputs = [prefix + example[source_lang] for example in examples["translation"]]
    targets = [example[target_lang] for example in examples["translation"]]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=256, truncation=True)
    return model_inputs

tokenized_books = train_data.map(preprocess_function, batched=True)

from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)

import evaluate

metric = evaluate.load("sacrebleu")

import numpy as np


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

training_args = Seq2SeqTrainingArguments(
    output_dir="my_awesome_opus_books_model",
    eval_strategy="steps",
    eval_steps=100,
    save_steps=100,
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=5,
    warmup_steps=500,
    logging_steps=50,
    predict_with_generate=True,
    load_best_model_at_end=True,
    metric_for_best_model="bleu",
    greater_is_better=True
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_books,
    eval_dataset=val_data.map(preprocess_function, batched=True),
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

# # Save the model and tokenizer locally
print("\nSaving model and tokenizer locally...")
trainer.save_model("my_awesome_opus_books_model")
tokenizer.save_pretrained("my_awesome_opus_books_model")

text = "translate Amharic to English: ከሚከተሉት እቃዎች ውስጥ ቁሳዊ ያልሆነ ባህል ምሳሌ የሚሆነው የትኛው ነው"
from transformers import pipeline

# Create translation pipeline using the local model
translator = pipeline("translation_amh_to_en", model="my_awesome_opus_books_model")
result = translator(text)
print("\nTranslation result:")
print(result[0]['translation_text'])
