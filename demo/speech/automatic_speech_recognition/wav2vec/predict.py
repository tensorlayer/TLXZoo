from tlxzoo.datasets import DataLoaders
from tlxzoo.module.wav2vec2 import Wav2Vec2Transform
from tlxzoo.speech.automatic_speech_recognition import AutomaticSpeechRecognition
import tensorlayerx as tlx
import numpy as np
import os


def device_info():
    found = False
    if not found and os.system("npu-smi info > /dev/null 2>&1") == 0:
        cmd = "npu-smi info"
        found = True
    elif not found and os.system("nvidia-smi > /dev/null 2>&1") == 0:
        cmd = "nvidia-smi"
        found = True
    elif not found and os.system("ixsmi > /dev/null 2>&1") == 0:
        cmd = "ixsmi"
        found = True
    elif not found and os.system("cnmon > /dev/null 2>&1") == 0:
        cmd = "cnmon"
        found = True
    
    os.system(cmd)
    cmd = "lscpu"
    os.system(cmd)
    
if __name__ == "__main__":
    device_info()
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