from tlxzoo.text.text_classification import TextClassification
from tlxzoo.module.bert.transform import BertTransform
import tensorlayerx as tlx


if __name__ == '__main__':
    transform = BertTransform(vocab_file="./demo/text/text_classification/bert/vocab.txt", max_length=128)
    model = TextClassification("bert")
    model.load_weights("./demo/text/text_classification/bert/model.npz")
    model.set_eval()

    text = "it 's a charming and often affecting journey ."
    x, y = transform(text, None)

    inputs = tlx.convert_to_tensor([x["inputs"]])
    token_type_ids = tlx.convert_to_tensor([x["token_type_ids"]])
    attention_mask = tlx.convert_to_tensor([x["attention_mask"]])

    _logits = model(inputs=inputs, token_type_ids=token_type_ids, attention_mask=attention_mask)
    label = tlx.argmax(_logits, axis=-1)
    print(label)

