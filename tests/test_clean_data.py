import unittest
import pandas as pd
from classification.data import clean_data


class CleanDataCase(unittest.TestCase):
    csv_file = 'data/dev_data.csv'

    def test_clean_data(self):
        df = pd.read_csv(self.csv_file)
        cleaned_df = clean_data(df, 'post_text', 'cate')
        self.assertNotEqual(''.join(df['post_text'].tolist()), ''.join(cleaned_df['text'].tolist()))
        print(cleaned_df)


if __name__ == '__main__':
    unittest.main()
