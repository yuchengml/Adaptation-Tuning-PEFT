import fire

from classification.lora import train_peft_model_w_lora
from classification.p_tuning import train_peft_model_w_p_tuning
from classification.prefix_tuning import train_peft_model_w_prefix_tuning
from classification.train import train_automodel


def fit_automodel_to_dev():
    train_automodel('data/dev_data_converted.csv')


def fit_automodel_to_post():
    train_automodel('data/posts.csv')


def fit_peft_w_prefix_tuning_to_dev():
    train_peft_model_w_prefix_tuning('data/dev_data_converted.csv')


def fit_peft_w_prefix_tuning_to_post():
    train_peft_model_w_prefix_tuning('data/posts.csv')


def fit_peft_w_p_tuning_to_dev():
    train_peft_model_w_p_tuning('data/dev_data_converted.csv')


def fit_peft_w_p_tuning_to_post():
    train_peft_model_w_p_tuning('data/posts.csv')


def fit_peft_w_lora_to_dev():
    train_peft_model_w_lora('data/dev_data_converted.csv')


def fit_peft_w_lora_to_post():
    train_peft_model_w_lora('data/posts.csv')


if __name__ == '__main__':
    fire.Fire({
        'fit_automodel_to_dev': fit_automodel_to_dev,
        'fit_automodel_to_post': fit_automodel_to_post,
        'fit_peft_w_prefix_tuning_to_dev': fit_peft_w_prefix_tuning_to_dev,
        'fit_peft_w_prefix_tuning_to_post': fit_peft_w_prefix_tuning_to_post,
        'fit_peft_w_p_tuning_to_dev': fit_peft_w_p_tuning_to_dev,
        'fit_peft_w_p_tuning_to_post': fit_peft_w_p_tuning_to_post,
        'fit_peft_w_lora_to_dev': fit_peft_w_lora_to_dev,
        'fit_peft_w_lora_to_post': fit_peft_w_lora_to_post
    })
