import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from data_provider.data_loader import Dataset_Custom, _explicit_date_borders


class ExplicitDateSplitTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='B')
        base = np.arange(len(dates), dtype=float)
        self.frame = pd.DataFrame({
            'date': dates.strftime('%Y-%m-%d'),
            'Open': base + 1.0,
            'High': base + 2.0,
            'Low': base,
            'Close': base + 1.5,
        })
        self.frame.to_csv(self.root / 'synthetic.csv', index=False)
        self.args = SimpleNamespace(
            split_mode='dates',
            train_end='2022-01-01',
            val_end='2023-01-01',
            test_end='2024-01-01',
            augmentation_ratio=0,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def make_dataset(self, flag):
        return Dataset_Custom(
            self.args,
            str(self.root),
            flag=flag,
            size=[20, 10, 5],
            features='MS',
            data_path='synthetic.csv',
            target='Close',
            scale=True,
            timeenc=0,
            freq='b',
        )

    def test_forecast_labels_stay_inside_declared_partition(self):
        expected = {
            'train': ('2020-01-29', '2021-12-31'),
            'val': ('2022-01-03', '2022-12-30'),
            'test': ('2023-01-02', '2023-12-29'),
        }
        for flag, (lower, upper) in expected.items():
            dataset = self.make_dataset(flag)
            self.assertGreater(len(dataset), 0)
            self.assertEqual(dataset.forecast_start_dates[0], pd.Timestamp(lower))
            self.assertEqual(dataset.forecast_end_dates[-1], pd.Timestamp(upper))
            if flag == 'val':
                self.assertGreaterEqual(dataset.forecast_start_dates.min(), pd.Timestamp('2022-01-01'))
                self.assertLess(dataset.forecast_end_dates.max(), pd.Timestamp('2023-01-01'))
            if flag == 'test':
                self.assertGreaterEqual(dataset.forecast_start_dates.min(), pd.Timestamp('2023-01-01'))
                self.assertLess(dataset.forecast_end_dates.max(), pd.Timestamp('2024-01-01'))

    def test_scaler_is_fit_only_on_training_rows(self):
        dataset = self.make_dataset('test')
        train = self.frame[pd.to_datetime(self.frame['date']) < pd.Timestamp('2022-01-01')]
        expected_mean = train[['Open', 'High', 'Low', 'Close']].to_numpy().mean(axis=0)
        np.testing.assert_allclose(dataset.scaler.mean_, expected_mean)

    def test_invalid_dates_and_too_short_partitions_fail_closed(self):
        with self.assertRaisesRegex(ValueError, 'train_end < val_end < test_end'):
            _explicit_date_borders(
                self.frame['date'], 20, 5, '2023-01-01', '2022-01-01', '2024-01-01'
            )
        with self.assertRaisesRegex(ValueError, 'validation partition'):
            _explicit_date_borders(
                self.frame['date'], 20, 10, '2022-01-01', '2022-01-05', '2024-01-01'
            )

    def test_legacy_ratio_mode_remains_available(self):
        ratio_args = SimpleNamespace(split_mode='ratio', augmentation_ratio=0)
        dataset = Dataset_Custom(
            ratio_args,
            str(self.root),
            flag='test',
            size=[20, 10, 5],
            features='MS',
            data_path='synthetic.csv',
            target='Close',
            scale=True,
            timeenc=0,
            freq='b',
        )
        self.assertEqual(dataset.split_metadata['mode'], 'ratio')
        self.assertGreater(len(dataset), 0)


if __name__ == '__main__':
    unittest.main()
