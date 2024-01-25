import unittest
from classification.data import prepare_dataset


class PrepareDatasetCase(unittest.TestCase):
    train_csv = '../data/dev_data_converted.csv'
    test_csv = '../data/dev_data_converted.csv'

    def test_prepare_dataset(self):
        ds, label_dict = prepare_dataset(self.train_csv, self.test_csv)
        self.assertEqual(ds['train'].shape[0], 4)
        self.assertEqual(ds['test'].shape[0], 4)

    def test_split_data(self):
        ds, label_dict = prepare_dataset(self.train_csv, test_size=0.2)
        self.assertEqual(ds['train'].shape[0], 3)
        self.assertEqual(ds['test'].shape[0], 1)


if __name__ == '__main__':
    unittest.main()
