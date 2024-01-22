import pandas as pd
import datasets

from classification.filters import InvalidCharacterFilter, URLFilter
from classification.utils import get_logger
import csv

logger = get_logger()

_labels = ['運動', '藝術', '交通', '服飾', '金融', '建築', '科技', '旅遊',
           '遊戲', '美食', '親子', '醫療', '教育', '寵物', '娛樂']


def clean_data(
        df: pd.DataFrame,
        text_col: str = "post_text",
        label_col: str = "cate") -> pd.DataFrame:
    # Simple preprocess
    df = df.dropna(subset=[text_col, label_col]).drop_duplicates(subset=[text_col, label_col])

    # Filter: invalid characters
    invalid_char_filter = InvalidCharacterFilter()
    # Filter: URL
    url_filter = URLFilter()

    posts = df[text_col].tolist()
    posts = list(map(lambda x: url_filter(invalid_char_filter(x.replace("\n", " "))), posts))

    labels = df[label_col].tolist()
    labels = list(map(lambda x: url_filter(invalid_char_filter(x)), labels))

    new_df = pd.DataFrame({"text": posts, "label": labels})
    return new_df


def view_data_cate(
        df: pd.DataFrame,
        cate_col: str = "label"):
    g = df.groupby(cate_col)
    dt = dict()
    for group in g.groups:
        g_df = g.get_group(group)
        dt[group] = len(g_df)
    return dt


def prepare_dataset(
        train_csv_path: str = None,
        test_csv_path: str = None,
        label_col: str = "label"):
    """Prepare transformers dataset from csv


    Args:
        train_csv_path:
            CSV path to load raw data
        test_csv_path:
            Partition data in csv as testing data
        label_col:
            Label column name
    Returns:

    """

    # Prepare label dictionary to map labels and indices
    logger.info("Prepare label dictionary")
    label_dict = dict(zip(_labels, list(range(len(_labels)))))

    logger.info("Prepare training and testing data")
    # Fetch training data
    df = pd.read_csv(train_csv_path, quoting=csv.QUOTE_ALL)

    df = clean_data(df)

    # Fetch testing data
    test_df = pd.read_csv(test_csv_path, quoting=csv.QUOTE_ALL)

    test_df = clean_data(test_df)

    logger.info(f"Train data: {view_data_cate(df)}")
    logger.info(f"Test data: {view_data_cate(test_df)}")

    # Convert categories to indices
    df[label_col] = df[label_col].map(lambda l: label_dict[l])
    test_df[label_col] = test_df[label_col].map(lambda l: label_dict[l])

    # Create datasets
    train_ds = datasets.Dataset.from_pandas(df).shuffle(seed=42)
    test_ds = datasets.Dataset.from_pandas(test_df).shuffle(seed=42)
    ds = datasets.DatasetDict({"train": train_ds, "test": test_ds})

    return ds, label_dict
