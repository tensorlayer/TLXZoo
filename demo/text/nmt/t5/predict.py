from tlxzoo.module.t5 import T5Transform
import tensorlayerx as tlx
from tlxzoo.text.text_conditional_generation import TextForConditionalGeneration


if __name__ == '__main__':
    model = TextForConditionalGeneration("t5")
    model.load_weights("./demo/text/nmt/t5/model.npz")
    model.set_eval()

    transform = T5Transform(vocab_file="./demo/text/nmt/t5/spiece.model", source_max_length=128, label_max_length=128)

    text = "Plane giants often trade blows on technical matters through advertising in the trade press."

    x, y = transform(text, "")

    inputs = tlx.convert_to_tensor([x["inputs"]], dtype=tlx.int64)
    attention_mask = tlx.convert_to_tensor([x["attention_mask"]], dtype=tlx.int64)

    decode_id = model.generate_one(inputs=inputs, attention_mask=attention_mask)
    decode_str = transform.ids_to_string(decode_id[0])

    print(decode_str)
