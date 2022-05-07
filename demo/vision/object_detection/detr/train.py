from tlxzoo.datasets import DataLoaders
from tlxzoo.module.detr import DetrTransform, post_process
from tlxzoo.vision.object_detection import ObjectDetection
from tlxzoo.datasets.coco import CocoEvaluator
import tensorlayerx as tlx


def valid(model, data_loaders):
    from tqdm import tqdm
    coco_evaluator = CocoEvaluator(data_loaders.dataset_dict["test"].coco, "bbox")
    model.set_eval()

    for idx, batch in enumerate(tqdm(data_loaders.test)):
        inputs = batch[0]["inputs"]
        pixel_mask = batch[0]["pixel_mask"]
        labels = batch[1]
        # forward pass
        outputs = model(inputs=inputs)

        orig_target_sizes = tlx.convert_to_tensor([target["orig_size"] for target in batch[1]], dtype=tlx.float32)

        results = post_process(outputs["pred_logits"], outputs["pred_boxes"], orig_target_sizes)

        res = {target['image_id']: output for target, output in zip(labels, results)}
        coco_evaluator.update(res)

    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()


class Trainer(tlx.model.Model):

    def tf_train(
            self, n_epoch, train_dataset, network, loss_fn, train_weights, optimizer, metrics, print_train_batch,
            print_freq, test_dataset
    ):
        import tensorflow as tf
        import time
        for epoch in range(n_epoch):
            start_time = time.time()

            train_loss, train_acc, n_iter = 0, 0, 0
            for X_batch, y_batch in train_dataset:
                network.set_train()

                with tf.GradientTape() as tape:
                    # compute outputs
                    _logits = network(**X_batch)
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


if __name__ == '__main__':
    transform = DetrTransform()
    # download coco from https://cocodataset.org/#download
    coco = DataLoaders("Coco", per_device_train_batch_size=1, per_device_eval_batch_size=1,
                       root_path="./adhub/coco2017/0.1",
                       train_ann_path="./adhub/coco2017/0.1/annotations/instances_train2017.json",
                       val_ann_path="./adhub/coco2017/0.1/annotations/instances_val2017.json",
                       num_workers=0,
                       collate_fn=transform.collate_fn,
                       train_limit=10,
                       )

    coco.register_transform_hook(transform)

    model = ObjectDetection(backbone="detr")

    model.load_weights("demo/vision/object_detection/detr/model.npz")

    optimizer = tlx.optimizers.Adam(lr=1e-6)
    metric = None

    trainer = Trainer(network=model, loss_fn=model.loss_fn, optimizer=optimizer, metrics=metric)
    trainer.train(n_epoch=1, train_dataset=coco.train, test_dataset=coco.test, print_freq=1,
                  print_train_batch=True)

    valid(model, coco)



