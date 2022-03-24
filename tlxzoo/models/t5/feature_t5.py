from ...feature.feature import BaseTextFeature, Trie
from ...config.config import BaseTextFeatureConfig
import sentencepiece as spm
from ...utils.registry import Registers
import re
import numpy as np


@Registers.feature_configs.register
class T5FeatureConfig(BaseTextFeatureConfig):
    def __init__(self,
                 vocab_file=None,
                 eos_token="</s>",
                 unk_token="<unk>",
                 pad_token="<pad>",
                 extra_ids=100,
                 prefix="translate English to French: ",
                 next_prefix=None,
                 task="text",
                 source_max_length=512,
                 label_max_length=512,
                 **kwargs):
        self.vocab_file = vocab_file
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.extra_ids = extra_ids
        self.source_max_length = source_max_length
        self.prefix = prefix
        self.task = task
        self.next_prefix = next_prefix
        self.label_max_length = label_max_length
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

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        if index < self.sp_model.get_piece_size():
            token = self.sp_model.IdToPiece(index)
        else:
            token = f"<extra_id_{self.vocab_size - 1 - index}>"
        return token

    def convert_tokens_to_string(self, tokens, remove_special_token=False):
        """Converts a sequence of tokens (string) in a single string."""
        current_sub_tokens = []
        out_string = ""
        for token in tokens:
            # make sure that special tokens are not decoded using sentencepiece model
            if token in self.additional_special_tokens:
                if not remove_special_token:
                    out_string += self.sp_model.decode_pieces(current_sub_tokens) + token + " "
                    current_sub_tokens = []
            else:
                current_sub_tokens.append(token)
        out_string += self.sp_model.decode_pieces(current_sub_tokens)
        return out_string.strip()

    def ids_to_string(self, ids, remove_special_token=True):
        tokens = [self._convert_id_to_token(int(index)) for index in ids if index >= 0]
        return self.convert_tokens_to_string(tokens, remove_special_token=remove_special_token)

    def string_to_ids(self, text, max_length=None):
        if isinstance(text, list):
            input_ids = []
            attention_masks = []
            for i in text:
                input_id, attention_mask = self.string_to_ids(i, max_length=max_length)
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
                    tokens = tokens + [self.config.eos_token] + [self.config.pad_token] * (
                            max_length - tokens_length - 1)

        ids = self.convert_tokens_to_ids(tokens)
        return {"inputs": np.array(ids), "attention_mask": np.array(attention_mask)}

    def process_token(self, text, label, source_max_length=None):
        ids = []
        labels = []
        for token, l in zip(text, label):
            token = self.tokenize(token)
            id = self.convert_tokens_to_ids(token)
            l = [l] * len(id)
            ids += id
            labels += l

        if source_max_length is None:
            ids = ids + self.convert_tokens_to_ids([self.config.eos_token])
            attention_mask = [1] * len(ids)
            labels = labels + [-100]
        else:
            if not isinstance(source_max_length, int):
                raise ValueError(f"{source_max_length} is not int.")
            else:
                ids_length = len(ids)
                if ids_length >= (source_max_length - 1):
                    ids = ids[:source_max_length - 1] + self.convert_tokens_to_ids([self.config.eos_token])
                    attention_mask = [1] * len(ids)
                    labels = labels[:source_max_length - 1] + [-100]
                else:
                    attention_mask = [1] * (len(ids) + 1) + [0] * (source_max_length - ids_length - 1)
                    ids = ids + self.convert_tokens_to_ids([self.config.eos_token]) + self.convert_tokens_to_ids(
                        [self.config.pad_token]) * (source_max_length - ids_length - 1)
                    labels = labels + [-100] * (source_max_length - ids_length)
        return {"inputs": np.array(ids), "attention_mask": np.array(attention_mask), "labels": np.array(labels)}, \
               {"labels": np.array(labels)}

    def __call__(self, text, label, source_max_length=None, label_max_length=None):
        if self.config.task == "token":
            return self.process_token(text, label,
                                      source_max_length if source_max_length else self.config.source_max_length)
        if isinstance(text, str):
            if self.config.prefix:
                text = self.config.prefix + text
            inputs = self.string_to_ids(text,
                                        max_length=source_max_length if source_max_length else self.config.source_max_length)
        elif (isinstance(text, tuple) or isinstance(text, list)) and len(text) == 2:
            first_text = text[0]
            if self.config.prefix:
                first_text = self.config.prefix + first_text
            first_inputs = self.string_to_ids(first_text,
                                              max_length=source_max_length if source_max_length else self.config.source_max_length)

            second_text = text[1]
            if self.config.next_prefix:
                second_text = self.config.next_prefix + second_text
            second_inputs = self.string_to_ids(second_text,
                                               max_length=source_max_length if source_max_length else self.config.source_max_length)
            inputs = {"inputs": np.concatenate([first_inputs["inputs"], second_inputs["inputs"]]),
                      "attention_mask": np.concatenate(
                          [first_inputs["attention_mask"], second_inputs["attention_mask"]])}
        else:
            raise ValueError(f"{text} is wrong.")

        if isinstance(label, str):
            labels = self.string_to_ids(label,
                                        max_length=label_max_length if label_max_length else self.config.label_max_length)
            labels = np.where(labels["attention_mask"], labels["inputs"], -100)
        else:
            labels = label
        labels = {"labels": labels}
        inputs.update(labels)
        return inputs, labels
