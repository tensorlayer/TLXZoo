import json
import numpy as np
import tensorlayerx as tlx
from itertools import groupby
from collections import OrderedDict
from tensorlayerx import logging


class Trie:
    """
    Trie in Python. Creates a Trie out of a list of words. The trie is used to split on `added_tokens` in one pass
    Loose reference https://en.wikipedia.org/wiki/Trie
    """

    def __init__(self):
        self.data = {}

    def add(self, word: str):
        """
        Passes over every char (utf-8 char) on word and recursively adds it to the internal `data` trie representation.
        The special key `""` is used to represent termination.

        This function is idempotent, adding twice the same word will leave the trie unchanged

        Example::

            >>> trie = Trie()
            >>> trie.add("Hello 友達")
            >>> trie.data
            {"H": {"e": {"l": {"l": {"o": {" ": {"友": {"達": {"": 1}}}}}}}}}
            >>> trie.add("Hello")
            >>> trie.data
            {"H": {"e": {"l": {"l": {"o": {"": 1, " ": {"友": {"達": {"": 1}}}}}}}}}
        """
        if not word:
            # Prevent empty string
            return
        ref = self.data
        for char in word:
            ref[char] = char in ref and ref[char] or {}
            ref = ref[char]
        ref[""] = 1

    def split(self, text: str):
        """
        Will look for the words added to the trie within `text`. Output is the original string splitted along the
        boundaries of the words found.

        This trie will match the longest possible word first !

        Example::

            >>> trie = Trie()
            >>> trie.split("[CLS] This is a extra_id_100")
            ["[CLS] This is a extra_id_100"]
            >>> trie.add("[CLS]")
            >>> trie.add("extra_id_1")
            >>> trie.add("extra_id_100")
            >>> trie.split("[CLS] This is a extra_id_100")
            ["[CLS]", " This is a ", "extra_id_100"]
        """
        # indexes are counted left of the chars index.
        # "hello", index 0, is left of h, index 1 is between h and e.
        # index 5 is right of the "o".

        # States are going to capture every possible start (indexes as above)
        # as keys, and have as values, a pointer to the position in the trie
        # where we're at. This is a partial match for now.
        # This enables to keep track of multiple matches while we're iterating
        # the string
        # If the trie contains, "blowing", and "lower" and we encounter the
        # string "blower", we need to split into ["b", "lower"].
        # This is where we need to keep track of multiple possible starts.
        states = OrderedDict()

        # This will contain every indices where we need
        # to cut.
        # We force to cut at offset 0 and len(text) (added later)
        offsets = [0]

        # This is used by the lookahead which needs to skip over
        # some text where the full match exceeded the place in the initial
        # for loop
        skip = None
        # Main loop, Giving this algorithm O(n) complexity
        for current, current_char in enumerate(text):
            if skip and current < skip:
                # Prevents the lookahead for matching twice
                # like extra_id_100 and id_100
                continue

            # This will track every state
            # that stop matching, we need to stop tracking them.
            # If we look at "lowball", we're going to match "l" (add it to states), "o", "w", then
            # fail on "b", we need to remove 0 from the valid states.
            to_remove = set()
            # Whenever we found a match, we need to drop everything
            # this is a greedy algorithm, it will match on the first found token
            reset = False

            # In this case, we already have partial matches (But unfinished)
            for start, trie_pointer in states.items():
                if "" in trie_pointer:
                    # This is a final match, we need to reset and
                    # store the results in `offsets`.

                    # Lookahead to match longest first
                    # Important in case of extra_id_1 vs extra_id_100
                    # Here we are also actively looking for other earlier partial
                    # matches
                    # "[CLS]", "L", we need to match CLS even if L is special
                    for lookstart, looktrie_pointer in states.items():
                        if lookstart > start:
                            # This partial match is later, we can stop looking
                            break
                        elif lookstart < start:
                            # This partial match is earlier, the trie pointer
                            # was already updated, so index is + 1
                            lookahead_index = current + 1
                            end = current + 1
                        else:
                            # Here lookstart == start and
                            #      looktrie_pointer == trie_pointer
                            # It wasn't updated yet so indices are current ones
                            lookahead_index = current
                            end = current
                        next_char = text[lookahead_index] if lookahead_index < len(text) else None
                        while next_char in looktrie_pointer:
                            looktrie_pointer = looktrie_pointer[next_char]
                            lookahead_index += 1
                            if "" in looktrie_pointer:
                                start = lookstart
                                end = lookahead_index
                                skip = lookahead_index

                            if lookahead_index == len(text):
                                # End of string
                                break
                            next_char = text[lookahead_index]
                        # End lookahead

                    # Storing and resetting
                    offsets.append(start)
                    offsets.append(end)
                    reset = True
                    break
                elif current_char in trie_pointer:
                    # The current character being looked at has a match within the trie
                    # update the pointer (it will be stored back into states later).
                    trie_pointer = trie_pointer[current_char]

                    # Storing back the new pointer into the states.
                    # Partial matches got longer by one.
                    states[start] = trie_pointer
                else:
                    # The new character has not match in the trie, we need
                    # to stop keeping track of this partial match.
                    # We can't do it directly within the loop because of how
                    # python iteration works
                    to_remove.add(start)

            # Either clearing the full start (we found a real match)
            # Or clearing only the partial matches that didn't work.
            if reset:
                states = {}
            else:
                for start in to_remove:
                    del states[start]

            # If this character is a starting character within the trie
            # start keeping track of this partial match.
            if current_char in self.data:
                states[current] = self.data[current_char]

        # We have a cut at the end with states.
        for start, trie_pointer in states.items():
            if "" in trie_pointer:
                # This is a final match, we need to reset and
                # store the results in `offsets`.
                end = len(text)
                offsets.append(start)
                offsets.append(end)
                # Longest cut is always the one with lower start so the first
                # item so we need to break.
                break

        return self.cut_text(text, offsets)

    def cut_text(self, text, offsets):
        # We have all the offsets now, we just need to do the actual splitting.
        # We need to eventually add the first part of the string and the eventual
        # last part.
        offsets.append(len(text))
        tokens = []
        start = 0
        for end in offsets:
            if start > end:
                logging.error(
                    "There was a bug in Trie algorithm in tokenization. Attempting to recover. Please report it anyway."
                )
                continue
            elif start == end:
                # This might happen if there's a match at index 0
                # we're also preventing zero-width cuts in case of two
                # consecutive matches
                continue
            tokens.append(text[start:end])
            start = end

        return tokens


class Wav2Vec2Transform(object):
    def __init__(
            self,
            vocab_file,
            feature_size=1,
            sampling_rate=16000,
            padding_value=0.0,
            return_attention_mask=False,
            do_normalize=True,
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            pad_token="<pad>",
            word_delimiter_token="|",
            do_lower_case=False,
            speech_max_length=None,
            text_max_length=None,
            **kwargs
    ):
        self.vocab_file = vocab_file
        self.feature_size = feature_size
        self.sampling_rate = sampling_rate
        self.padding_value = padding_value
        self.return_attention_mask = return_attention_mask
        self.do_normalize = do_normalize
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.word_delimiter_token = word_delimiter_token
        self.do_lower_case = do_lower_case
        self.speech_max_length = speech_max_length
        self.text_max_length = text_max_length
        self.is_train = True
        super(Wav2Vec2Transform, self).__init__(**kwargs)

        with open(self.vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        self.decoder = {v: k for k, v in self.encoder.items()}

        self.unique_no_split_tokens = []
        for token in self.encoder.keys():
            if len(token) > 1:
                self.unique_no_split_tokens.append(token)

        self._create_trie(self.unique_no_split_tokens)

    def set_eval(self):
        self.is_train = False

    def set_train(self):
        self.is_train = True

    def _create_trie(self, unique_no_split_tokens):
        trie = Trie()
        for token in unique_no_split_tokens:
            if hasattr(self, "do_lower_case") and self.do_lower_case and token not in self.all_special_tokens:
                trie.add(token.lower())
            else:
                trie.add(token)
        self.tokens_trie = trie

    def tokenize(self, text, **kwargs):
        """
        Converts a string in a sequence of tokens (string), using the tokenizer.
        """
        if self.do_lower_case:
            text = text.upper()

        return list(text.replace(" ", self.word_delimiter_token))

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an index (integer) using the vocab."""
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        result = self.decoder.get(index, self.unk_token)
        return result

    def process_speech(self, speech):
        if self.do_normalize:
            input_values = (speech - speech.mean()) / np.sqrt(speech.var() + 1e-7)
        else:
            input_values = speech
        return input_values

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

    def string_to_ids(self, text):
        tokens = self.tokenize(text)
        ids = self.convert_tokens_to_ids(tokens)
        return ids

    def __call__(self, speech, text, *args, **kwargs):
        speech = speech.astype(np.float32)
        input_values = self.process_speech(speech)
        # input_values = input_values.astype(np.float32)
        input_ids = self.string_to_ids(text)
        input_ids = np.array(input_ids, dtype=np.int32)
        # return {"input_values": input_values}, {"input_ids": input_ids}
        return input_values, input_ids

    def pad_and_create_pixel_mask(
        self, pixel_values_list
    ):

        max_size = max([speech.shape[0] for speech in pixel_values_list])
        if self.speech_max_length:
            if max_size > self.speech_max_length:
                max_size = self.speech_max_length
        padded_pixel_values = []
        pixel_mask = []
        for pixel_value in pixel_values_list:

            padded_pixel_value = np.zeros(max_size, dtype=np.float32)
            mask = np.zeros(max_size, dtype=bool)
            if pixel_value.shape[0] > max_size:
                padded_pixel_value = pixel_value[:max_size]
                mask = np.ones(max_size, dtype=bool)
            else:
                padded_pixel_value[: pixel_value.shape[0]] = np.copy(pixel_value)
                mask[: pixel_value.shape[0]] = True

            padded_pixel_values.append(padded_pixel_value)
            mask = mask.astype(np.int32)
            pixel_mask.append(mask)

        return padded_pixel_values, pixel_mask

    def pad_ids(self, ids):
        max_size = max([label.shape[0] for label in ids])
        if self.text_max_length:
            if max_size > self.text_max_length:
                max_size = self.text_max_length

        padded_ids = []
        for label in ids:
            padded_id = np.zeros(max_size, dtype=np.int32)
            if label.shape[0] > max_size:
                padded_id = label[:max_size]
            else:
                padded_id[: label.shape[0]] = np.copy(label)
            padded_ids.append(padded_id)

        return padded_ids

    def collate_fn(self, data):
        input_values = [i[0] for i in data]
        padded_pixel_values, pixel_mask = self.pad_and_create_pixel_mask(input_values)

        # inputs = {"inputs": padded_pixel_values, "pixel_mask": pixel_mask}

        labels = [i[1][0] for i in data]
        texts = [i[1][1] for i in data]
        padded_ids = self.pad_ids(labels)
        # labels = {"labels": padded_ids}

        if len(data) >= 2:
            new_data = []
            for i, j, k, l in zip(padded_pixel_values, pixel_mask, padded_ids, texts):
                new_data.append(({"inputs": i, "pixel_mask": j}, {"labels": k, "texts": l}))

            return tlx.dataflow.dataloader.utils.default_collate(new_data)
        else:
            return tlx.dataflow.dataloader.utils.default_convert(({"inputs": np.array(padded_pixel_values),
                                                                   "pixel_mask": np.array(pixel_mask)},
                                                                  {"labels": np.array(padded_ids), "texts": texts}))

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        if isinstance(ids, int):
            return self._convert_id_to_token(ids)
        tokens = []
        for index in ids:
            index = int(index)
            if skip_special_tokens and index in self.all_special_ids:
                continue
            tokens.append(self._convert_id_to_token(index))
        return tokens

    def convert_tokens_to_string(
            self, tokens, group_tokens=True, spaces_between_special_tokens= False):
        """
        Converts a connectionist-temporal-classification (CTC) output tokens into a single string.
        """
        # group same tokens into non-repeating tokens in CTC style decoding
        if group_tokens:
            tokens = [token_group[0] for token_group in groupby(tokens)]

        # filter self.pad_token which is used as CTC-blank token
        filtered_tokens = list(filter(lambda token: token != self.pad_token, tokens))

        if spaces_between_special_tokens:
            join_token = " "
        else:
            join_token = ""

        # replace delimiter token
        string = join_token.join(
            [" " if token == self.word_delimiter_token else token for token in filtered_tokens]
        ).strip()

        if self.do_lower_case:
            string = string.lower()
        return string

    @staticmethod
    def clean_up_tokenization(out_string: str) -> str:
        """
        Clean up a list of simple English tokenization artifacts like spaces before punctuations and abbreviated forms.

        Args:
            out_string (:obj:`str`): The text to clean up.

        Returns:
            :obj:`str`: The cleaned-up string.
        """
        out_string = (
            out_string.replace(" .", ".")
                .replace(" ?", "?")
                .replace(" !", "!")
                .replace(" ,", ",")
                .replace(" ' ", "'")
                .replace(" n't", "n't")
                .replace(" 'm", "'m")
                .replace(" 's", "'s")
                .replace(" 've", "'ve")
                .replace(" 're", "'re")
        )
        return out_string

    def ids_to_string(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True,
                      group_tokens=True, spaces_between_special_tokens=False):
        filtered_tokens = self.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens)

        result = []
        for token in filtered_tokens:
            if skip_special_tokens and token in self.all_special_ids:
                continue
            result.append(token)

        text = self.convert_tokens_to_string(
            result, group_tokens=group_tokens, spaces_between_special_tokens=spaces_between_special_tokens
        )

        if clean_up_tokenization_spaces:
            clean_text = self.clean_up_tokenization(text)
            return clean_text
        else:
            return text






