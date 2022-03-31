from tlxzoo.dataset import DataLoaders, ImageDetectionDataConfig
import tensorlayerx as tlx

from tlxzoo.models.detr import *
from tlxzoo.dataset.coco import CocoEvaluator
import time
from tqdm import tqdm

feat_config = DetrFeatureConfig()
feat = DetrFeature(feat_config)

image_detection_config = ImageDetectionDataConfig(root_path="/home/xiaolong-xu/adhub/coco2017/0.1",
                                                  per_device_eval_batch_size=1,
                                                  per_device_train_batch_size=1,
                                                  train_ann_path="/home/xiaolong-xu/adhub/coco2017/0.1/annotations/instances_train2017.json",
                                                  val_ann_path="/home/xiaolong-xu/adhub/coco2017/0.1/annotations/instances_val2017.json",
                                                  num_workers=0)

data_loaders = DataLoaders(image_detection_config, train_limit=3, collate_fn=feat.collate_fn)

data_loaders.register_transform_hook(feat)

detr_model_config = DetrModelConfig()
detr_task_config = DetrForObjectDetectionTaskConfig(detr_model_config, auxiliary_loss=True, weights_path="detr")


"""
download weight from
WEIGHT_NAME_TO_CKPT = {
    "detr": [
        "https://storage.googleapis.com/visualbehavior-publicweights/detr/checkpoint",
        "https://storage.googleapis.com/visualbehavior-publicweights/detr/detr.ckpt.data-00000-of-00001",
        "https://storage.googleapis.com/visualbehavior-publicweights/detr/detr.ckpt.index"
    ]
}
"""
detr = DetrForObjectDetection.from_pretrained(config=detr_task_config, format="ckpt",
                                              pretrained_base_path="/home/xiaolong-xu/userdata/tensorlayerX/detr-tensorflow/weights")


def valid(model):
    coco_evaluator = CocoEvaluator(data_loaders.dataset_dict["test"].coco, "bbox")
    model.set_eval()

    for idx, batch in enumerate(tqdm(data_loaders.test)):
        # print(batch)
        inputs = batch[0]["inputs"]
        pixel_mask = batch[0]["pixel_mask"]
        labels = batch[1]
        # forward pass
        outputs = model(inputs=inputs, return_output=True)

        orig_target_sizes = tlx.convert_to_tensor([target["orig_size"] for target in batch[1]], dtype=tlx.float32)

        results = post_process(outputs["pred_logits"], outputs["pred_boxes"], orig_target_sizes)

        res = {target['image_id']: output for target, output in zip(labels, results)}
        coco_evaluator.update(res)

    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()


# valid(detr)


class Trainer(tlx.model.Model):

    def tf_train(
            self, n_epoch, train_dataset, network, loss_fn, train_weights, optimizer, metrics, print_train_batch,
            print_freq, test_dataset
    ):
        import tensorflow as tf
        for epoch in range(n_epoch):
            start_time = time.time()

            train_loss, train_acc, n_iter = 0, 0, 0
            for X_batch, y_batch in train_dataset:
                network.set_train()

                with tf.GradientTape() as tape:
                    # compute outputs
                    _logits = network(**X_batch, return_output=True)
                    # _loss_ce = tf.reduce_mean(loss_fn(_logits, y_batch))
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

        valid(network)


optimizer = tlx.optimizers.Adam(learning_rate=1e-6)
metric = None

model = Trainer(network=detr, loss_fn=detr.loss_fn, optimizer=optimizer, metrics=metric)
model.train(n_epoch=1, train_dataset=data_loaders.train, test_dataset=data_loaders.test, print_freq=1,
            print_train_batch=True)




