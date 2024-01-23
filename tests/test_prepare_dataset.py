import unittest
from classification.data import prepare_dataset


class PrepareDatasetCase(unittest.TestCase):
    train_csv = '../data/dev_data_converted.csv'
    test_csv = '../data/dev_data_converted.csv'

    def test_prepare_dataset(self):
        ds, label_dict = prepare_dataset(self.train_csv, self.test_csv)


if __name__ == '__main__':
    unittest.main()
