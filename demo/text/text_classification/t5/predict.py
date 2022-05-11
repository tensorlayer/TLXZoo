from tlxzoo.datasets import DataLoaders
from tlxzoo.module.t5 import T5Transform
import tensorlayerx as tlx
from tlxzoo.text.text_classification import TextClassification


if __name__ == '__main__':
    transform = T5Transform(vocab_file="./demo/text/nmt/t5/spiece.model", prefix="sst2 sentence: ",
                            source_max_length=256)
    model = TextClassification("t5")
    model.load_weights("./demo/text/text_classification/t5/model.npz")
    model.set_eval()

    text = "it 's a charming and often affecting journey ."
    x, y = transform(text, None)

    inputs = tlx.convert_to_tensor([x["inputs"]])
    attention_mask = tlx.convert_to_tensor([x["attention_mask"]])

    _logits = model(inputs=inputs, attention_mask=attention_mask)
    label = tlx.argmax(_logits, axis=-1)
    print(label)

