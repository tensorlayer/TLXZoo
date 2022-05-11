from tlxzoo.module.bert.transform import BertTransform
import tensorlayerx as tlx
from tlxzoo.text.text_token_classidication import TextTokenClassification


if __name__ == '__main__':
    transform = BertTransform(vocab_file="./demo/text/text_classification/bert/vocab.txt", task="token", max_length=128)

    model = TextTokenClassification("bert", n_class=9)
    model.load_weights("./demo/text/token_classification/bert/model.npz")

    tokens = ["CRICKET", "-", "LEICESTERSHIRE", "TAKE", "OVER", "AT", "TOP", "AFTER", "INNINGS", "VICTORY", "."]
    labels = ["O", "O", "B-ORG", "O", "O", "O", "O", "O", "O", "O", "O"]

    x, y = transform(tokens, labels)

    inputs = tlx.convert_to_tensor([x["inputs"]])
    token_type_ids = tlx.convert_to_tensor([x["token_type_ids"]])
    attention_mask = tlx.convert_to_tensor([x["attention_mask"]])

    _logits = model(inputs=inputs, token_type_ids=token_type_ids, attention_mask=attention_mask)

    labels = tlx.argmax(_logits, axis=-1)
    print(y)
    print(labels)
