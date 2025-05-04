from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer
from datasets import Dataset

def prepare_dataset(df, tokenizer_name="distilbert-base-uncased"):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def tokenize(batch):
        return tokenizer(batch['empathetic_dialogues'], truncation=True, padding=True)

    dataset = Dataset.from_pandas(df[['empathetic_dialogues', 'emotion_label']])
    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'emotion_label'])
    return dataset, tokenizer

def train_model(dataset, model_name="distilbert-base-uncased", output_dir="./models"):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=32)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        logging_dir='./logs',
        evaluation_strategy='epoch',
        save_strategy='epoch'
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset.select(range(500))
    )

    trainer.train()
    model.save_pretrained(output_dir)
    return model
