from tlxzoo.datasets import DataLoaders
from tlxzoo.module.wav2vec2 import Wav2Vec2Transform
from tlxzoo.speech.automatic_speech_recognition import AutomaticSpeechRecognition
import tensorlayerx as tlx
import numpy as np


if __name__ == '__main__':
    transform = Wav2Vec2Transform(vocab_file="./demo/speech/automatic_speech_recognition/wav2vec/vocab.json")

    model = AutomaticSpeechRecognition(backbone="wav2vec")
    model.load_weights("./demo/speech/automatic_speech_recognition/wav2vec/model.npz")

    import soundfile as sf

    file = "./demo/speech/automatic_speech_recognition/wav2vec/1272-128104-0000.flac"
    speech, _ = sf.read(file)

    input_values, input_ids = transform(speech, "")
    mask = np.ones(input_values.shape[0], dtype=np.int32)

    input_values = tlx.convert_to_tensor([input_values])
    pixel_mask = tlx.convert_to_tensor([mask])

    logits = model(inputs=input_values, pixel_mask=pixel_mask)
    predicted_ids = tlx.argmax(logits, axis=-1)

    transcription = transform.ids_to_string(predicted_ids[0])

    print(transcription)