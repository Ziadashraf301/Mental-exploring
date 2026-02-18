from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from Depression_detection.src.logger.train_logger import get_logger
from Depression_detection.src.config.train_config_loader import get_train_config
from Depression_detection.src.utils.metrics_utils import compute_metrics_torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
import torch

LOGGER = get_logger()

# MULTINOMIAL NAIVE BAYES
def train_multinomial_nb(X, y):
    """
    Train Multinomial Naive Bayes using config parameters.
    """
    config = get_train_config()
    params = config.pipelines["classical_ml"]["models"]["multinomial_nb"]

    model = MultinomialNB(
        alpha=params.get("alpha", 1.0),
        fit_prior=params.get("fit_prior", False),
        force_alpha=params.get("force_alpha", True)
    )

    model.fit(X, y)
    LOGGER.info(f"Trained MultinomialNB with params: {params}")

    return model


# SGD Classifier
def train_sgd_classifier(X, y):
    """
    Train Linear SVC using config parameters.
    """
    config = get_train_config()
    params = config.pipelines["classical_ml"]["models"]["sgd_classifier"]

    model = SGDClassifier(
        loss=params.get("loss", "modified_huber"),
        penalty=params.get("penalty", "l2"),
        alpha=params.get("alpha", 0.0001),
        max_iter=params.get("max_iter", 60),
        tol=params.get("tol", None),
        learning_rate=params.get("learning_rate", "adaptive"),
        eta0=params.get("eta0", 0.01),
        fit_intercept=params.get("fit_intercept", False),
        random_state=config.random_seeds.get("numpy_seed", 42)
    )

    model.fit(X, y)
    LOGGER.info(f"Trained SGD Classifier with params: {params}")

    return model


def run_cv_sklearn(model, X, y, scoring: str):
    """Run K-Fold CV only if enabled in config."""
    
    config = get_train_config()
    
    LOGGER.info(
        f"Running {config.cv_params.get('k_folds', 5)}-fold cross-validation "
        f"(shuffle={config.cv_params.get('shuffle', True)}, random_state={config.cv_params.get('random_state', 42)})"
        f" with scoring='{scoring}'"
    )

    cv_scores = cross_val_score(
        model,
        X,
        y,
        shuffle=config.cv_params.get("shuffle", True),
        random_state=config.cv_params.get("random_state", 42),
        cv=config.cv_params.get("k_folds", 5),
        scoring=scoring,
        n_jobs=config.cv_params.get("n_jobs", -1)
    )

    LOGGER.info(f"CV Mean {scoring}: {cv_scores.mean():.4f} | Std: {cv_scores.std():.4f}")
    return cv_scores.tolist()


def train_bert_model_task(train_dataset, val_dataset, output_dir):
    config = get_train_config()
    bert_cfg = config.pipelines["bert"]
    training_cfg = bert_cfg["training"]
    lora_cfg = bert_cfg["lora"]
    tokenization_cfg = bert_cfg["tokenization"]

    # Load base model
    model = AutoModelForSequenceClassification.from_pretrained(
        tokenization_cfg["model_name"],
        num_labels=2,
        problem_type="single_label_classification"
    )

    # Apply LoRA if enabled
    if lora_cfg["enabled"]:
        lora_config = LoraConfig(
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["lora_alpha"],
            target_modules=lora_cfg["target_modules"],
            lora_dropout=lora_cfg["lora_dropout"],
            bias=lora_cfg["bias"],
            task_type="SEQ_CLS" 
        )

        model = get_peft_model(model, lora_config)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=training_cfg["num_epochs"],
        per_device_train_batch_size=training_cfg["batch_size"],
        per_device_eval_batch_size=training_cfg["batch_size"],
        warmup_steps=training_cfg["warmup_steps"],
        weight_decay=training_cfg["weight_decay"],
        logging_dir=f"{output_dir}/logs",
        logging_steps=training_cfg["logging_steps"],
        eval_strategy="steps",
        eval_steps=training_cfg["eval_steps"],
        save_strategy="steps",
        save_steps=training_cfg["save_steps"],
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=training_cfg["save_total_limit"],
        fp16=training_cfg["fp16"] and torch.cuda.is_available(),
        report_to=training_cfg["report_to"],
        seed=config.random_seeds.get("global", 42)
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_torch,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=training_cfg["early_stopping_patience"]
        )]
    )

    # Train
    train_result = trainer.train()

    return model, trainer