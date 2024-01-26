from typing import List

import evaluate
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

from classification.utils import get_logger


def compute_metrics(eval_pred):
    """Compute accuracy"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)
    metric = evaluate.load('accuracy')
    return metric.compute(predictions=predictions, references=labels)
