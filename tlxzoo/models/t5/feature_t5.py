from ...feature.feature import BaseTextFeature, Trie
from ...config.config import BaseTextFeatureConfig
import sentencepiece as spm
from ...utils.registry import Registers
import re
import numpy as np


class T5FeatureConfig(BaseTextFeatureConfig):
    def __init__(self,
                 vocab_file=None,
                 eos_token="</s>",
                 unk_token="<unk>",
                 pad_token="<pad>",
                 extra_ids=100,
                 max_length=512,
                 **kwargs):

        self.vocab_file = vocab_file
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.extra_ids = extra_ids
        self.max_length = max_length
        super(T5FeatureConfig, self).__init__(**kwargs)


@Registers.features.register
class T5Feature(BaseTextFeature):
    config_class = BaseTextFeatureConfig

    def __init__(
            self,
            config,
            **kwargs
    ):
        self.config = config
        if self.config.extra_ids > 0:
            self.additional_special_tokens = [f"<extra_id_{i}>" for i in range(self.config.extra_ids)]

        if self.config.eos_token is not None:
            self.additional_special_tokens.append(self.config.eos_token)

        if self.config.unk_token is not None:
            self.additional_special_tokens.append(self.config.unk_token)

        if self.config.pad_token is not None:
            self.additional_special_tokens.append(self.config.pad_token)

        self._create_trie(self.additional_special_tokens)

        if self.config.vocab_file is None:
            raise ValueError(f"vocab file is None.")
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(self.config.vocab_file)
        super(T5Feature, self).__init__(config, **kwargs)

    @property
    def vocab_size(self):
        return self.sp_model.get_piece_size() + self.config.extra_ids

    def _create_trie(self, unique_no_split_tokens):
        trie = Trie()
        for token in unique_no_split_tokens:
            if hasattr(self, "do_lower_case") and self.do_lower_case and token not in self.all_special_tokens:
                trie.add(token.lower())
            else:
                trie.add(token)
        self.tokens_trie = trie

    def _tokenize(self, token):
        return self.sp_model.encode(token, out_type=str)

    def tokenize(self, text):
        no_split_token = set(self.additional_special_tokens)
        tokens = self.tokens_trie.split(text)

        # ["This is something", "<special_token_1>", "  else"]
        for i, token in enumerate(tokens):
            if token in no_split_token:
                left = tokens[i - 1] if i > 0 else None
                right = tokens[i + 1] if i < len(tokens) - 1 else None
                # We strip left and right by default
                if right:
                    tokens[i + 1] = right.lstrip()
                if left:
                    tokens[i - 1] = left.rstrip()
        # ["This is something", "<special_token_1>", "else"]
        tokenized_text = []
        for token in tokens:
            # Need to skip eventual empty (fully stripped) tokens
            if not token:
                continue
            if token in no_split_token:
                tokenized_text.append(token)
            else:
                tokenized_text.extend(self._tokenize(token))
        # ["This", " is", " something", "<special_token_1>", "else"]
        return tokenized_text

    def convert_tokens_to_ids(self, tokens):
        if tokens is None:
            return None

        if isinstance(tokens, str):
            return self._convert_token_to_id_with_added_voc(tokens)

        ids = []
        for token in tokens:
            ids.append(self._convert_token_to_id_with_added_voc(token))
        return ids

    def _convert_token_to_id_with_added_voc(self, token):
        if token is None:
            return None

        return self._convert_token_to_id(token)

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        if token.startswith("<extra_id_"):
            match = re.match(r"<extra_id_(\d+)>", token)
            num = int(match.group(1))
            return self.vocab_size - num - 1
        return self.sp_model.piece_to_id(token)

    def __call__(self, text, max_length=None):
        if max_length is None:
            max_length = self.config.max_length
        if isinstance(text, list):
            input_ids = []
            attention_masks = []
            for i in text:
                input_id, attention_mask = self.__call__(i, max_length=max_length)
                input_ids.append(input_id)
                attention_masks.append(attention_mask)
            return {"inputs": np.array(input_ids), "attention_mask": np.array(attention_masks)}
        tokens = self.tokenize(text)

        if max_length is None:
            tokens = tokens + [self.config.eos_token]
            attention_mask = [1] * len(tokens)
        else:
            if not isinstance(max_length, int):
                raise ValueError(f"{max_length} is not int.")
            else:
                tokens_length = len(tokens)
                if tokens_length >= (max_length - 1):
                    tokens = tokens[:max_length - 1] + [self.config.eos_token]
                    attention_mask = [1] * len(tokens)
                else:
                    attention_mask = [1] * (len(tokens) + 1) + [0] * (max_length - tokens_length - 1)
                    tokens = tokens + [self.config.eos_token] + [self.config.pad_token] * (max_length - tokens_length - 1)

        ids = self.convert_tokens_to_ids(tokens)
        return {"inputs": np.array(ids), "attention_mask": np.array(attention_mask)}

