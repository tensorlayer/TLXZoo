import tensorlayerx as tlx
from tensorlayerx.vision.transforms import Compose
from tensorlayerx.dataflow import DataLoader
from tlxzoo.datasets import Face300WDataset
from tlxzoo.module.pfld import *
from tlxzoo.vision.facial_landmark_detection import FacialLandmarkDetection


if __name__ == '__main__':
    transform = Compose([
        Crop(),
        Resize(size=(112, 112)),
        RandomHorizontalFlip(),
        RandomRotate(angle_range=list(range(-30, 31, 5))),
        RandomOcclude(occlude_size=(50, 50)),
        Normalize(),
        CalculateEulerAngles(),
        ToTuple()
    ])
    train_dataset = Face300WDataset('./data/300W', split='train', transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=2)
    test_dataset = Face300WDataset('./data/300W', split='test', transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=2)

    model = FacialLandmarkDetection(backbone="pfld")

    optimizer = tlx.optimizers.Adam(1e-4, weight_decay=1e-6)
    metrics = NME()
    n_epoch = 500

    trainer = tlx.model.Model(
        network=model, loss_fn=model.loss_fn, optimizer=optimizer, metrics=metrics)
    trainer.train(n_epoch=n_epoch, train_dataset=train_dataloader,
                  test_dataset=test_dataloader, print_freq=1, print_train_batch=False)

    model.save_weights("./demo/vision/facial_landmark_detection/pfld/model.npz")
