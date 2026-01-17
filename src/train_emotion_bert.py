import json
import os
import random

import evaluate
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

# Configuration for BERT multilabel classification
MODEL_NAME = "bert-base-uncased"
DATASET = "dair-ai/emotion"
DATA_CONFIG = "split"
MAX_LENGTH = 128
BATCH_SIZE = 16
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
OUTPUT_DIR = "./bert-emotion"
SEED = 42
NUM_LABELS = 6
LABEL_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]
LATEST_CHECKPOINT = "./bert-emotion/checkpoint-2000"


def set_seed(seed: int = SEED) -> bool:
    """Use SEED config value to guarantee reproducibility.

    Returns:
        bool: True after completion.
    """

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    return True


def load_ds(dataset: str = DATASET, data_config: str = DATA_CONFIG) -> dict:
    """Load dataset from HuggingFace to use for training and evaluation.

    Args:
        dataset (str, optional): Name of dataset. Defaults to DATASET.
        data_config (str, optional): Parameter to select configuration to
            download. Defaults to DATA_CONFIG.

    Returns:
        dict: Dataset to use for training and evaluation.
    """

    return load_dataset(dataset, data_config)


def create_tokenizer(model_name: str = MODEL_NAME) -> AutoTokenizer:
    """Create AutoTokenizer to use for BERT classification.

    Args:
        model_name (str, optional): Name of model to use. Defaults to
            MODEL_NAME.

    Returns:
        AutoTokenizer: Instance of transformer-based AutoTokenizer.
    """

    return AutoTokenizer.from_pretrained(model_name)


def preprocess_function(tokenizer: AutoTokenizer, data: dict) -> dict:
    """Embed the text and add labels.

    Args:
        tokenizer (AutoTokenizer): Transformer tokenizer to use for embedding.
        data (dict): Data containing labels and text.

    Returns:
        dict: Text that has been tokenized and embedded with labels added.
    """

    enc = tokenizer(
        data["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )
    enc["labels"] = data["label"]
    return enc


def create_model(
    model_name: str = MODEL_NAME, num_labels: int = NUM_LABELS
) -> AutoModelForSequenceClassification:
    return AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )


def create_collator(tokenizer) -> DataCollatorWithPadding:
    return DataCollatorWithPadding(tokenizer=tokenizer)


def train_eval(model, tokenizer, collator, tokenized):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy.compute(predictions=preds, references=labels)[
                "accuracy"
            ],
            "f1_macro": f1.compute(
                predictions=preds, references=labels, average="macro"
            )["f1"],
            "precision_macro": precision.compute(
                predictions=preds, references=labels, average="macro"
            )["precision"],
            "recall_macro": recall.compute(
                predictions=preds, references=labels, average="macro"
            )["recall"],
        }

    # Define metrics for model evaluation
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    # Configure arguments for training
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=2,
        seed=SEED,
        fp16=False,
    )

    # Configure Trainer instance on tokenized dataset
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    print("Training classifier...")

    # Train model, strating from the latest checkpoint if available
    if LATEST_CHECKPOINT:
        trainer.train(LATEST_CHECKPOINT)
    else:
        trainer.train()

    # Evaluate on test set
    metrics = trainer.evaluate(tokenized["test"])
    print("Test metrics:", metrics)

    # Save trained model
    trainer.save_model(OUTPUT_DIR)

    # Save label mapping
    with open(os.path.join(OUTPUT_DIR, "label_mapping.json"), "w") as f:
        json.dump({i: LABEL_NAMES[i] for i in range(NUM_LABELS)}, f)


def main() -> None:
    """Main function to execute all functions."""

    set_seed()
    ds = load_ds()
    tokenizer = create_tokenizer()
    tokenized = ds.map(
        lambda batch: preprocess_function(tokenizer, batch),
        batched=True,
        remove_columns=["text"],
    )

    model = create_model()
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_eval(model, tokenizer, collator, tokenized)


if __name__ == "__main__":
    main()
