import fire
from classification.train import train_automodel


def fit_automodel_to_dev():
    train_automodel(
        'data/dev_data_converted.csv',
        'data/dev_data_converted.csv',
    )


if __name__ == '__main__':
    fire.Fire({
        'fit_automodel_to_dev': fit_automodel_to_dev
    })
