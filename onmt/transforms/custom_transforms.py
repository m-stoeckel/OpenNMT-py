from itertools import zip_longest
from string import punctuation, whitespace
from typing import List

import regex as rx
import unicodedata

from onmt.transforms import register_transform, Transform


@register_transform(name='normalize')
class Normalize(Transform):
    """
    Normalize input text by applying Unicode normalization, and replacing unconventional quotes with regular ones.
    """

    @classmethod
    def add_options(cls, parser):
        """Avalilable options relate to this Transform."""
        group = parser.add_argument_group("Transform/Normalize")
        group.add("--no_normalize_quotes", action='store_true', default=False,
                  help="Disable quote normalization.")
        group.add("--no_normalize_contractions", action='store_true', default=False,
                  help="Disable contraction normalization.")
        group.add("--no_unicode_normalization", action='store_true', default=False,
                  help="Disable unicode normalization.")
        group.add("--unicode_normalization_form", default='NFKC', choices=['NFC', 'NFKC', 'NFD', 'NFKD'],
                  help="The unicode normalization form. Default 'NFKC'.")

    def _parse_opts(self):
        self.no_normalize_quotes = self.opts.no_normalize_quotes
        self.no_normalize_contractions = self.opts.no_normalize_contractions
        self.no_unicode_normalization = self.opts.no_unicode_normalization
        self.unicode_normalization_form = self.opts.unicode_normalization_form

    def _normalize(self, sentence: List[str]) -> List[str]:
        sentence = ' '.join(sentence)

        # Normalize quotes
        if not self.no_normalize_quotes:
            sentence = sentence.replace("\u00AB", '"') \
                .replace("\u2039", '"') \
                .replace("\u00BB", '"') \
                .replace("\u203A", '"') \
                .replace("\u201E", '"') \
                .replace("\u201C", '"') \
                .replace("\u201F", '"') \
                .replace("\u201D", '"') \
                .replace("\u0022", '"') \
                .replace("\u275D", '"') \
                .replace("\u275E", '"') \
                .replace("\u276E", '"') \
                .replace("\u276F", '"') \
                .replace("\u2E42", '"') \
                .replace("\u301D", '"') \
                .replace("\u301E", '"') \
                .replace("\u301F", '"') \
                .replace("\uFF02", '"') \
                .replace("\u201A", '"') \
                .replace("\u2018", '"') \
                .replace("\u201B", '"') \
                .replace("\u275B", '"') \
                .replace("\u275C", '"') \
                .replace("\u275F", '"')

        # Normalize contractions
        if not self.no_normalize_contractions:
            sentence = sentence.replace(r'’', r"'")

        # Normalize Unicode symbols and accents
        if not self.no_unicode_normalization:
            sentence = unicodedata.normalize(self.unicode_normalization_form, sentence)

        return sentence.strip().split()

    def apply(self, example, is_train=False, stats=None, **kwargs):
        example['src'] = self._normalize(example['src'])
        example['tgt'] = self._normalize(example['tgt'])
        return example

    def _repr_args(self):
        """Return str represent key arguments for class."""
        return ", ".join([
            f"no_normalize_quotes={self.no_normalize_quotes}",
            f"no_normalize_contractions={self.no_normalize_contractions}",
            f"no_unicode_normalization={self.no_unicode_normalization}",
            f"unicode_normalization_form={self.unicode_normalization_form}",
        ])


@register_transform(name='pre_tokenize')
class PreTokenize(Transform):
    """Pre-tokenize sentences, separating whitespaces from words while leaving intra-word whitespaces intact."""

    @classmethod
    def add_options(cls, parser):
        """Avalilable options relate to this Transform."""
        group = parser.add_argument_group("Transform/Normalize")
        group.add("--intra_word_punctuation", default=".',-",
                  help="Intra-word punctuation characters. Defaults to: '.\',-'")
        group.add("--sticky_punctuation", default='.',
                  help="Punctuation characters that 'stick' to the end of words (e.g. in case of abbreviations). "
                       "Note: Make sure you only add the period character '.' here if your data is properly segmented "
                       "into sentences. Defaults to: '.'")

    def _parse_opts(self):
        self.inter_word_punctuation = set(punctuation).difference(self.opts.intra_word_punctuation)
        self.non_sticky_punctuation = set(punctuation).difference(self.opts.sticky_punctuation)
        self.whitespace_or_punctuation = set(whitespace + punctuation)

    def _pretok_gen(self, sentence: str):
        prev_char = ' '
        for char, next_char in zip_longest(sentence, sentence[1:], fillvalue='\0'):
            char_in_inter_word_punctuation = char in self.inter_word_punctuation
            char_in_non_sticky_punctuation = char in self.non_sticky_punctuation
            char_eq_next_char = char == next_char
            if prev_char != char and (
                    char_in_inter_word_punctuation or (
                        char_in_non_sticky_punctuation and next_char in self.whitespace_or_punctuation
                    ) or (
                        char in punctuation and char_eq_next_char
                    ) or next_char == '\0'
            ):
                yield ' '
            yield char
            if not char_eq_next_char and (
                    char_in_inter_word_punctuation or (
                        char_in_non_sticky_punctuation and prev_char in self.whitespace_or_punctuation
                    )
            ):
                yield ' '
            prev_char = char

    def _pre_tokenize(self, sentence: List[str]) -> List[str]:
        sentence = ' '.join(sentence)

        # Split punctuation marks from words
        sentence = ''.join(self._pretok_gen(sentence))

        return sentence.strip().split()

    def apply(self, example, is_train=False, stats=None, **kwargs):
        example['src'] = self._pre_tokenize(example['src'])
        example['tgt'] = self._pre_tokenize(example['tgt'])
        return example


@register_transform(name='clean_ted')
class CleanTED(Transform):
    """Remove audience reactions from TEDx talk transcripts."""

    remove_bracket = rx.compile(r"\([^)]+\)")

    def clean(self, sentence: List[str]) -> List[str]:
        sentence = ' '.join(sentence)
        if '(' in sentence and ')' in sentence:
            sentence = self.remove_bracket.sub('', sentence)
        if '♫' in sentence:
            sentence = sentence.replace('♫ ', '').replace('♫', '')
        return sentence.strip().split()

    def apply(self, example, is_train=False, stats=None, **kwargs):
        example['src'] = self.clean(example['src'])
        example['tgt'] = self.clean(example['tgt'])
        return example
