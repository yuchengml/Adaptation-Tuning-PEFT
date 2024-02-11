import datetime
import os.path
from typing import Dict, Any

from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding

from classification.data import prepare_dataset
from classification.metrics import compute_metrics
from classification.utils import get_logger

logger = get_logger()


def train_peft_model_w_lora(
        train_csv_path: str,
        test_csv_path: str = None,
        pretrain_model: str = "prebuilt_model/chinese-roberta-wwm-ext-large",
        output_dir: str = "prebuilt_model",
        text_col: str = "text",
        label_col: str = "label",
        batch_size: int = 8,
        n_epochs: int = 10,
        use_wandb: bool = True,
):
    """Train a classification model using transformers. with LoRA in PEFT.
     LoRA decomposes a large matrix into two smaller low-rank matrices in the attention layers.
     (Read https://arxiv.org/abs/2309.15223 to learn more about LoRA.)

    Args:
        train_csv_path:
            CSV path to load raw data.
        test_csv_path:
            Partition data in csv as testing data.  If `None`, split data from training.
        pretrain_model:
            Assign transformers pretrained model path or name.
        output_dir:
            The output directory where the model predictions and checkpoints will be written.
        text_col:
            Text column name.
        label_col:
            Label column name.
        batch_size:
            Batch size to prepare data and model training.
        n_epochs:
            Assign training epochs.
        use_wandb:
            Determine to use wandb to track training metrics.

    Returns:

    """
    # Prepare transformers dataset from csv(raw data)
    ds, label_dict = prepare_dataset(
        train_csv_path=train_csv_path,
        test_csv_path=test_csv_path,
        label_col=label_col)
    id2label = dict(zip(label_dict.values(), label_dict.keys()))
    label2id = label_dict
    n_labels = len(label_dict)

    if not os.path.exists(pretrain_model):
        pretrain_model = "hfl/chinese-roberta-wwm-ext-large"

    # Create tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(pretrain_model)
    except FileNotFoundError as e:
        logger.error(e)
        exit(1)

    # Define preprocess function to preprocess raw data
    def preprocess_function(examples: Dict[str, Any]):
        # return tokenizer(examples[text_col], truncation=True)
        return tokenizer(examples[text_col], truncation=True, max_length=512)

    # Do preprocessing
    tokenized_posts = ds.map(preprocess_function,
                             batch_size=batch_size,
                             writer_batch_size=batch_size,
                             batched=True)

    # Data collator that will dynamically pad the inputs received
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # LoRA Cconfig
    peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, inference_mode=False, r=16, lora_alpha=16, lora_dropout=0.1,
                             bias="all")

    # Create classification model
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrain_model, num_labels=n_labels, id2label=id2label, label2id=label2id)

    # Get Peft model object from a model and a config
    model = get_peft_model(model, peft_config)
    # model.print_trainable_parameters()
    trainable_params, all_param = model.get_nb_trainable_parameters()

    logger.info(
        f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}"
    )

    folder_name = f"{datetime.datetime.now().strftime('%y%m%d-%H%M%S')}"
    log_dir = os.path.join(output_dir, folder_name)

    # Use wandb to track training metrics
    report_args = []
    if use_wandb:
        import wandb
        wandb.init(project="adaptation-tuning-peft", name=folder_name, tags=['lora'])
        report_args.append("wandb")
        wandb.config["batch_size"] = batch_size
        wandb.config["trainable_params"] = trainable_params
        wandb.config["all_params"] = all_param

    # Create training arguments
    training_args = TrainingArguments(
        output_dir=log_dir,
        learning_rate=3e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=n_epochs,
        weight_decay=0.01,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        report_to=report_args,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        greater_is_better=True
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_posts["train"],
        eval_dataset=tokenized_posts["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Execute training
    trainer.train()

    # Test prediction
    # print(trainer.predict(tokenized_posts["test"]))

    trainer.save_model(log_dir)
    trainer.save_model(os.path.join(output_dir, "classifier"))

    return folder_name
