from tlxzoo.dataset import DataLoaders, FaceRecognitionDataConfig
from tlxzoo.models.retinaface import *
from tlxzoo.models.retinaface.task_retinaface import load_ckpt_weight
import os
from tqdm import tqdm
import pathlib

feat_config = RetinaFaceFeatureConfig()
feat = RetinaFaceFeature(feat_config)

image_config = FaceRecognitionDataConfig(root_path="./widerface",
                                         per_device_eval_batch_size=1,
                                         per_device_train_batch_size=4,
                                         train_ann_path="label.txt",
                                         val_ann_path="label.txt",
                                         num_workers=2)

data_loaders = DataLoaders(image_config)

data_loaders.register_transform_hook(feat)

model_config = RetinaFaceModelConfig()
task_config = RetinaFaceForFaceRecognitionTaskConfig(model_config)

retina_face_model = RetinaFaceForFaceRecognition(task_config)
# download weight from https://github.com/peteryuX/retinaface-tf2
retina_face_model = load_ckpt_weight(retina_face_model, "./checkpoints/retinaface_res50")


def test_widerface(model):
    model.set_eval()
    feat.set_eval()
    for i, j in tqdm(data_loaders.test):
        output = model(i)
        outputs = feat.decode_one(output[0], output[1], output[2], i, j[1], score_th=0.5)

        img_path = j[2][0]
        img_name = os.path.basename(img_path)
        sub_dir = os.path.basename(os.path.dirname(img_path))
        save_folder = "./widerface"
        save_name = os.path.join(save_folder, sub_dir, img_name.replace('.jpg', '.txt'))
        pathlib.Path(os.path.join(save_folder, sub_dir)).mkdir(
            parents=True, exist_ok=True)

        img_raw = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img_height_raw, img_width_raw, _ = img_raw.shape

        with open(save_name, "w") as file:
            bboxs = outputs[:, :4]
            confs = outputs[:, -1]

            file_name = img_name + "\n"
            bboxs_num = str(len(bboxs)) + "\n"
            file.write(file_name)
            file.write(bboxs_num)
            for box, conf in zip(bboxs, confs):
                x = int(box[0] * img_width_raw)
                y = int(box[1] * img_height_raw)
                w = int(box[2] * img_width_raw) - int(box[0] * img_width_raw)
                h = int(box[3] * img_height_raw) - int(box[1] * img_height_raw)
                confidence = str(conf)
                line = str(x) + " " + str(y) + " " + str(w) + " " + str(h) \
                    + " " + confidence + " \n"
                file.write(line)

        for prior_index in range(len(outputs)):
            draw_bbox_landm(img_raw, outputs[prior_index],
                            img_height_raw, img_width_raw)
        os.makedirs(os.path.join(save_folder, "images"), exist_ok=True)
        cv2.imwrite(os.path.join(save_folder, "images", img_name), img_raw)


test_widerface(retina_face_model)


class Trainer(tlx.model.Model):
    def tf_train(
            self, n_epoch, train_dataset, network, loss_fn, train_weights, optimizer, metrics, print_train_batch,
            print_freq, test_dataset
    ):
        import tensorflow as tf
        import time
        for epoch in range(n_epoch):
            feat.set_train()
            start_time = time.time()

            train_loss, train_acc, n_iter = 0, 0, 0
            for X_batch, y_batch in train_dataset:
                network.set_train()

                with tf.GradientTape() as tape:
                    # compute outputs
                    _logits = network(X_batch)
                    _loss_ce = loss_fn(_logits, y_batch)

                grad = tape.gradient(_loss_ce, train_weights)

                optimizer.apply_gradients(zip(grad, train_weights))
                train_loss += _loss_ce

                n_iter += 1

                if print_train_batch:
                    print("Epoch {} of {} {} took {}".format(epoch + 1, n_epoch, n_iter, time.time() - start_time))
                    print("   train loss: {}".format(train_loss / n_iter))

            if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
                print("Epoch {} of {} took {}".format(epoch + 1, n_epoch, time.time() - start_time))
                print("   train loss: {}".format(train_loss / n_iter))

            test_widerface(network)


optimizer = tlx.optimizers.SGD(lr=4e-5, momentum=0.9, weight_decay=5e-4)
trainer = Trainer(network=retina_face_model, loss_fn=retina_face_model.loss_fn, optimizer=optimizer, metrics=None)
trainer.train(n_epoch=2, train_dataset=data_loaders.train, test_dataset=data_loaders.test, print_freq=1,
              print_train_batch=True)