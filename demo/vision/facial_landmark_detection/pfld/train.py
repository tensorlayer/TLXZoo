import tensorlayerx as tlx
from tlxzoo.datasets import DataLoaders
from tlxzoo.module.pfld import NME, Face300WTransform
from tlxzoo.vision.facial_landmark_detection import FacialLandmarkDetection


if __name__ == '__main__':
    transform = Face300WTransform()
    face300w = DataLoaders("Face300W", root_path='./data/300W',
                           per_device_train_batch_size=256, per_device_eval_batch_size=64)
    face300w.register_transform_hook(transform)

    model = FacialLandmarkDetection(backbone="pfld")

    optimizer = tlx.optimizers.Adam(1e-4, weight_decay=1e-6)
    metrics = NME()
    n_epoch = 500

    trainer = tlx.model.Model(
        network=model, loss_fn=model.loss_fn, optimizer=optimizer, metrics=metrics)
    trainer.train(n_epoch=n_epoch, train_dataset=face300w.train,
                  test_dataset=face300w.test, print_freq=1, print_train_batch=False)

    model.save_weights("./model.npz")
