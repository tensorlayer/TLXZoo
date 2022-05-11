from tlxzoo.module.trocr.transform import TrOCRTransform
from tlxzoo.vision.ocr import OpticalCharacterRecognition
import tensorlayerx as tlx


if __name__ == '__main__':
    transform = TrOCRTransform(merges_file="./demo/vision/ocr/trocr/merges.txt",
                               vocab_file="./demo/vision/ocr/trocr/vocab.json", max_length=12)

    model = OpticalCharacterRecognition("trocr")
    model.load_weights("./demo/vision/ocr/trocr/model.npz")
    model.set_eval()

    jpg_path = "./demo/vision/ocr/trocr/466_MONIKER_49537.jpg"

    x, y = transform(jpg_path, "")

    inputs = tlx.convert_to_tensor([x["inputs"]])
    predicted_ids = model.generate_one(inputs=inputs, max_length=24)
    transcription = transform.ids_to_string(predicted_ids[0])
    print(transcription)

