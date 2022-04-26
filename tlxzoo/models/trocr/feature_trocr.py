from ...feature.feature import BaseImageFeature
from ...config.config import BaseFeatureConfig
from ...utils.registry import Registers

from PIL import Image
import PIL.Image
import numpy as np
import cv2
import regex as re
import json
import tensorlayerx as tlx


@Registers.feature_configs.register
class TrOCRFeatureConfig(BaseFeatureConfig):
    def __init__(self,
                 vocab_file,
                 merges_file,
                 errors="replace",
                 bos_token="<s>",
                 eos_token="</s>",
                 sep_token="</s>",
                 cls_token="<s>",
                 unk_token="<unk>",
                 pad_token="<pad>",
                 mask_token="<mask>",
                 max_length=512,
                 do_resize=True,
                 size=384,
                 resample=2,
                 do_normalize=True,
                 image_mean=None,
                 image_std=None,
                 **kwargs):
        self.vocab_file = vocab_file
        self.merges_file = merges_file
        self.errors = errors
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.sep_token = sep_token
        self.cls_token = cls_token
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.mask_token = mask_token
        self.max_length = max_length

        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else [0.5, 0.5, 0.5]
        self.image_std = image_std if image_std is not None else [0.5, 0.5, 0.5]

        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        super(TrOCRFeatureConfig, self).__init__(**kwargs)


def get_pairs(word):
    """
    Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def bytes_to_unicode():
    bs = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


@Registers.features.register
class TrOCRFeature(BaseImageFeature):
    config_class = TrOCRFeatureConfig

    def __init__(
            self,
            config,
            **kwargs
    ):
        self.config = config
        self.cache = {}

        with open(self.config.merges_file, encoding="utf-8") as merges_handle:
            bpe_merges = merges_handle.read().split("\n")[1:-1]
        bpe_merges = [tuple(merge.split()) for merge in bpe_merges]
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))

        with open(self.config.vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        self.decoder = {v: k for k, v in self.encoder.items()}

        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        self.errors = "replace"
        self.special_tokens = [self.config.bos_token, self.config.eos_token, self.config.sep_token,
                               self.config.cls_token,
                               self.config.unk_token, self.config.pad_token, self.config.mask_token]

        super(TrOCRFeature, self).__init__(config, **kwargs)

    def resize(self, image, size, resample=PIL.Image.BILINEAR):
        if isinstance(size, int):
            size = (size, size)
        elif isinstance(size, list):
            size = tuple(size)
        return image.resize(size, resample=resample)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def tokenize(self, text):
        bpe_tokens = []
        for token in re.findall(self.config.pat, text):
            token = "".join(
                self.byte_encoder[b] for b in token.encode("utf-8")
            )  # Maps all our bytes to unicode strings, avoiding control tokens of the BPE (spaces in our case)
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

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
            tokens = [self.config.bos_token] + tokens + [self.config.eos_token]
            attention_mask = [1] * len(tokens)
        else:
            if not isinstance(max_length, int):
                raise ValueError(f"{max_length} is not int.")
            else:
                tokens_length = len(tokens)
                if tokens_length >= (max_length - 2):
                    tokens = [self.config.bos_token] + tokens[:max_length - 2] + [self.config.eos_token]
                    attention_mask = [1] * len(tokens)
                else:
                    attention_mask = [1] * (len(tokens) + 2) + [0] * (max_length - tokens_length - 2)
                    tokens = [self.config.bos_token] + tokens + [self.config.eos_token] + [self.config.pad_token] * (
                            max_length - tokens_length - 2)

        ids = self.convert_tokens_to_ids(tokens)
        return {"inputs": np.array(ids), "attention_mask": np.array(attention_mask)}

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.encoder.get(token, self.encoder.get(self.config.unk_token))

    def convert_ids_to_tokens(self, ids):
        tokens = [self._convert_id_to_token(int(index)) for index in ids if index >= 0]
        return tokens

    def ids_to_string(self, ids, remove_special_token=True):
        tokens = [self._convert_id_to_token(int(index)) for index in ids if index >= 0]
        if remove_special_token:
            tokens = [i for i in tokens if i not in self.special_tokens]
        return self.convert_tokens_to_string(tokens)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.decoder.get(index)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        text = "".join(tokens)
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)
        return text

    def convert_tokens_to_ids(self, tokens):
        if tokens is None:
            return None

        if isinstance(tokens, str):
            return self._convert_token_to_id(tokens)

        ids = []
        for token in tokens:
            ids.append(self._convert_token_to_id(token))
        return ids

    def process_image(
        self,
        image,
    ):
        if self.config.do_resize and self.config.size is not None:
            image = self.resize(image=image, size=self.config.size, resample=self.config.resample)
        if self.config.do_normalize:
            image = self.normalize(image=image, mean=self.config.image_mean, std=self.config.image_std)

        return image

    def to_numpy_array(self, image, rescale=None, channel_first=True):
        if isinstance(image, Image.Image):
            image = np.array(image)

        if rescale is None:
            rescale = isinstance(image.flat[0], np.integer)

        if rescale:
            image = image.astype(np.float32) / 255.0

        if channel_first and image.ndim == 3:
            image = image.transpose(2, 0, 1)

        return image

    def normalize(self, image, mean, std):
        image = self.to_numpy_array(image)

        if not isinstance(mean, np.ndarray):
            mean = np.array(mean).astype(image.dtype)
        if not isinstance(std, np.ndarray):
            std = np.array(std).astype(image.dtype)

        if image.ndim == 3 and image.shape[0] in [1, 3]:
            return (image - mean[:, None, None]) / std[:, None, None]
        else:
            return (image - mean) / std

    def __call__(self, image_path, text):
        image = Image.open(image_path).convert("RGB")
        image = self.process_image(image)

        labels = self.string_to_ids(text, max_length=self.config.max_length)

        return {"inputs": image}, labels

    def collate_fn(self, data):
        input_values = [i[0]["inputs"] for i in data]
        texts = [i[1][1] for i in data]
        input_ids = [i[1][0]["inputs"] for i in data]
        attention_mask = [i[1][0]["attention_mask"] for i in data]

        input_values = np.array(input_values)
        input_ids = np.array(input_ids)
        attention_mask = np.array(attention_mask)

        length = np.max(np.sum(attention_mask, -1))
        length = int(length)
        input_ids = input_ids[:, :length]
        attention_mask = attention_mask[:, :length]

        return tlx.dataflow.dataloader.utils.default_convert(({"inputs": input_values}, {"inputs": input_ids,
                                                                                         "attention_mask": attention_mask, "texts": texts}))


@Registers.features.register
class CV2TrOCRFeature(TrOCRFeature):
    def __call__(self, image_path, text):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.process_image(image)

        labels = self.string_to_ids(text, max_length=self.config.max_length)

        return {"inputs": image}, labels

    def resize(self, image, size, resample=None):
        if isinstance(size, int):
            size = (size, size)
        elif isinstance(size, list):
            size = tuple(size)
        return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
