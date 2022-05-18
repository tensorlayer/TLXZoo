FROM tensorflow/tensorflow:2.6.0-gpu
WORKDIR /root
RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends sudo vim git ssh
RUN pip install tensorlayerx==0.5.1
RUN pip install tqdm
RUN pip install rouge
RUN pip install pandas
RUN pip install jupyterlab
RUN pip install opencv-python
RUN pip install SoundFile
RUN apt-get install libsndfile1
RUN pip install sentencepiece
RUN pip install sacrebleu
RUN pip install rouge_score
RUN pip install pycocotools
RUN apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends libgl1-mesa-glx

