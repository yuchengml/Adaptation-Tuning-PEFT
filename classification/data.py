import pandas as pd

from classification.filters import InvalidCharacterFilter, URLFilter


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
