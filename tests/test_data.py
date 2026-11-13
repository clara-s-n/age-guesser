import tempfile
import unittest
from pathlib import Path
from PIL import Image
from age_guesser import categorize_age, scan_dataset


class DataTests(unittest.TestCase):
    def test_boundaries(self):
        for age, label in [(0, 0), (12, 0), (13, 1), (25, 1), (26, 2), (59, 2), (60, 3), (116, 3)]:
            self.assertEqual(categorize_age(age), label)
        for age in [-1, 117]:
            with self.assertRaises(ValueError):
                categorize_age(age)

    def test_scan_rejects_invalid_and_corrupted_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new('RGB', (10, 10)).save(root / '60_0_0_date.jpg')
            Image.new('RGB', (10, 10)).save(root / 'unknown.jpg')
            (root / '12_0_0_date.jpg').write_text('corrupt')
            rows, rejected = scan_dataset(root)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['label'], 3)
            self.assertEqual(len(rejected), 2)


if __name__ == '__main__':
    unittest.main()
