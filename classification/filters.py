import re
from abc import ABC, abstractmethod


def _repl_func(x):
    return " " * (x.end() - x.start())


class Filter(ABC):
    """Abstract class to define a filtering method"""

    def __call__(self, *args, **kwargs):
        return self.filter(*args, **kwargs)

    @abstractmethod
    def filter(self, *args, **kwargs):
        """Filter job with conditions."""
        pass


class InvalidCharacterFilter(Filter):
    def __init__(self):
        # Space: \u0020
        # Numbers: \u0030-\u0039
        # Alphabets: \u0041-\u005A\u0061-\u007A
        # Punctuation & Symbols: \u0021-\u002F\u003A-\u0040\u005B-\u0060\u007B-\u007E
        # Chinese Symbols: \uFF01-\uFF0F\uFF1A-\uFF20\uFF3B-\uFF40\uFF5B-\uFF65\u3001\u3002
        # CJK Unified Ideographs: \u4E00-\u9FFF
        # Japanese: \u3040-\u30FF\u31F0-\u31FF\uFF66-\uFF9C
        # Korean: \u1100-\u11FF\u3130-\u318F\uAC00-\uD7AF
        # German/French: \u00C0-\u00FF
        # New Line Chars(LF, NEL, LS, PS): \u000A\u000D\u2028\u2029
        valid_chars = (
            "[^\u0020"
            "\u0030-\u0039"
            "\u0041-\u005A\u0061-\u007A"
            "\u0021-\u002F\u003A-\u0040\u005B-\u0060\u007B-\u007E"
            "\uFF01-\uFF0F\uFF1A-\uFF20\uFF3B-\uFF40\uFF5B-\uFF65\u3001\u3002"
            "\u4E00-\u9FFF"
            "\u3040-\u30FF\u31F0-\u31FF\uFF66-\uFF9C"
            "\u1100-\u11FF\u3130-\u318F\uAC00-\uD7AF"
            "\u00C0-\u00FF"
            "\u000A\u000D\u2028\u2029]"
        )
        self.re_pattern = re.compile(valid_chars, re.UNICODE)

    def filter(self, text: str, repl: str = _repl_func):
        """Filter text with any conditions.

        Args:
            text: One text string
            repl: Replacement character

        Returns:
            Filtered text string
        """
        if not isinstance(type(text), str):
            TypeError("`text` should be a string")

        filtered = self.re_pattern.sub(repl, "{}".format(text))
        return filtered


class URLFilter(Filter):
    def __init__(self):
        regex = (
            r"((?:https?://)?(?:(?:www\.)?(?:[\da-z\.-]+)\.(?:[a-z]{2,6})|"
            r"(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)|"
            r"(?:(?:[0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|(?:[0-9a-fA-F]{1,4}:){1,7}:|"
            r"(?:[0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|(?:[0-9a-fA-F]{1,4}:){1,5}(?::[0-9a-fA-F]{1,4}){1,2}|"
            r"(?:[0-9a-fA-F]{1,4}:){1,4}(?::[0-9a-fA-F]{1,4}){1,3}|(?:[0-9a-fA-F]{1,4}:){1,3}"
            r"(?::[0-9a-fA-F]{1,4}){1,4}|(?:[0-9a-fA-F]{1,4}:){1,2}(?::[0-9a-fA-F]{1,4}){1,5}|"
            r"[0-9a-fA-F]{1,4}:(?:(?::[0-9a-fA-F]{1,4}){1,6})|:(?:(?::[0-9a-fA-F]{1,4}){1,7}|:)|"
            r"fe80:(?::[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(?:ffff(?::0{1,4}){0,1}:){0,1}"
            r"(?:(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){"
            r"0,1}[0-9])|(?:[0-9a-fA-F]{1,4}:){1,4}:(?:(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}"
            r"(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])))(?::[0-9]{1,4}|[1-5][0-9]{4}|6[0-4][0-9]{3}|65[0-4]"
            r"[0-9]{2}|655[0-2][0-9]|6553[0-5])?(?:/[\w\.-]*)*/?)\b"
        )
        self.re_pattern = re.compile(regex, re.UNICODE)

    def filter(self, text: str, repl: str = _repl_func) -> str:
        if not isinstance(type(text), str):
            TypeError("`text` should be a string")

        return self.re_pattern.sub(repl, text)
