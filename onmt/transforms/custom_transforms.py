from typing import List

import regex as rx
from tokenizers import normalizers
from tokenizers.normalizers import NFD, StripAccents, Strip

from onmt.transforms import register_transform, Transform


@register_transform(name='normalize')
class Normalize(Transform):
    """
    Normalize input text by applying NFD and StripAccents normalization methods from the `tokenizers` library,
    removing trailing whitespaces and replacing unconventional quotes with regular ones.
    """

    def __init__(self, opts):
        super(Normalize, self).__init__(opts)

    def _parse_opts(self):
        self.no_normalize_quotes = self.opts.no_normalize_quotes
        self.no_normalize_contractions = self.opts.no_normalize_contractions

        self.normalizer = normalizers.Sequence([NFD(), StripAccents(), Strip()])

    @classmethod
    def add_options(cls, parser):
        """Avalilable options relate to this Transform."""
        group = parser.add_argument_group("Transform/Normalize")
        group.add("--no_normalize_quotes", action='store_true', default=False,
                  help="Disable quote normalization.")
        group.add("--no_normalize_contractions", action='store_true', default=False,
                  help="Disable contraction normalization.")

    def _normalize(self, sentence: List[str]) -> List[str]:
        sentence = ' '.join(sentence)

        # Normalize quotes
        if not self.no_normalize_quotes:
            sentence = rx.sub(r'(?<!\pL)[‚‘]+([^‚‘]+)[‚‘]+(?!\pL)', '"\\1"', sentence, flags=rx.UNICODE)
            sentence = rx.sub(r'(?<!\pL)[“”]+([^“”]+)[“”]+(?!\pL)', '"\\1"', sentence, flags=rx.UNICODE)

        # Normalize contractions
        if not self.no_normalize_contractions:
            sentence = sentence.replace('’', "'")

        # Normalize Unicode symbols and accents
        sentence = self.normalizer.normalize_str(sentence)

        return sentence.split()

    def apply(self, example, is_train=False, stats=None, **kwargs):
        example['src'] = self._normalize(example['src'])
        example['tgt'] = self._normalize(example['tgt'])
        return example

    def _repr_args(self):
        """Return str represent key arguments for class."""
        return ", ".join([
            f"no_normalize_quotes={self.no_normalize_quotes}",
            f"no_normalize_contractions={self.no_normalize_contractions}"
        ])


@register_transform(name='pretokenize')
class PreTokenize(Transform):
    """Pre-tokenize sentences, separating whitespaces from words while leaving intra-word whitespaces intact."""

    def _pre_tokenize(self, sentence: List[str]) -> List[str]:
        sentence = ' '.join(sentence)
        # Split punctuation marks from words
        sentence = rx.sub(r'((?<= )[\pP\pS]+|[\pP\pS]+(?= )|(?<=^)[\pP\pS]+|[\pP\pS]+(?=$))', ' \\1 ', sentence)

        # Split punctuation marks from each other, except EOS markers.
        sentence = rx.sub(r'((?<=[\.\!\?,])[^\PP\.\!\?,]|[^\PP\.\!\?,](?=[\.\!\?,]))', ' \\1 ', sentence)

        # FIXME: Remove. Superseded by previous regex.
        # sentence = rx.sub(r'([\pP\pS](?=[^\PP\.\!\?])|(?>[^\PP\.\!\?])[\pP\pS])', ' \\1 ', sentence)

        # Replace all whitespaces with regular one
        sentence = rx.sub(r'\pZ+', ' ', sentence)

        return sentence.split()

    def apply(self, example, is_train=False, stats=None, **kwargs):
        example['src'] = self._pre_tokenize(example['src'])
        example['tgt'] = self._pre_tokenize(example['tgt'])
        return example
