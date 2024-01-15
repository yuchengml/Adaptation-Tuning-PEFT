import unittest
from classification.filters import InvalidCharacterFilter, URLFilter


class TestFilters(unittest.TestCase):
    def test_invalid_character_filter(self):
        valid_text = [
            'A',  # (uppercase letter)
            '5',  # (digit)
            '!',  # (exclamation mark)
            '/',  # (forward slash)
            ':',  # (colon)
            '[',  # (left square bracket)
            ']',  # (right square bracket)
            '一',  # (CJK ideograph)
            'あ',  # (Hiragana letter)
            '가',  # (Hangul syllable)
        ]
        invalid_text = '😀🤖🖐️👍❤️💯😈😡'
        valid_text = ''.join(valid_text)

        _filter = InvalidCharacterFilter()
        filtered_text = _filter(valid_text + invalid_text).strip()
        self.assertEqual(filtered_text, valid_text)

    def test_url_filter(self):
        text = '123https://123.com'
        corrected_text = '123'

        _filter = URLFilter()
        filtered_text = _filter(text).strip()
        self.assertEqual(filtered_text, corrected_text)  # add assertion here


if __name__ == '__main__':
    unittest.main()
