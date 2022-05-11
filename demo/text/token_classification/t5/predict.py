from tlxzoo.module.t5 import T5Transform
import tensorlayerx as tlx
from tlxzoo.text.text_token_classidication import TextTokenClassification


if __name__ == '__main__':
    transform = T5Transform(vocab_file="./demo/text/nmt/t5/spiece.model", prefix="", task="token",
                            source_max_length=256, label_max_length=256)

    model = TextTokenClassification("t5", n_class=9)
    model.load_weights("./demo/text/token_classification/t5/model.npz")

    tokens = ["CRICKET", "-", "LEICESTERSHIRE", "TAKE", "OVER", "AT", "TOP", "AFTER", "INNINGS", "VICTORY", "."]
    labels = ["O", "O", "B-ORG", "O", "O", "O", "O", "O", "O", "O", "O"]

    x, y = transform(tokens, labels)

    inputs = tlx.convert_to_tensor([x["inputs"]])
    attention_mask = tlx.convert_to_tensor([x["attention_mask"]])

    _logits = model(inputs=inputs, attention_mask=attention_mask)

    labels = tlx.argmax(_logits, axis=-1)
    print(y)
    print(labels)
