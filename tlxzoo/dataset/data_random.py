import tensorlayerx as tlx


class DataRandom(object):
    def __init__(self,
                 random_rotation_degrees=None,
                 random_shift=None,
                 random_flip_horizontal_prop=None,
                 random_crop_size=None,
                 ):
        if random_rotation_degrees:
            self.RandomRotation = tlx.vision.transforms.RandomRotation(degrees=random_rotation_degrees,
                                                                       interpolation='bilinear', expand=False,
                                                                       center=None, fill=0)
        else:
            self.RandomRotation = None

        if random_shift:
            self.RandomShift = tlx.vision.transforms.RandomShift(shift=random_shift, interpolation='bilinear', fill=0)
        else:
            self.RandomShift = None

        if random_flip_horizontal_prop:
            self.RandomFlipHorizontal = tlx.vision.transforms.RandomFlipHorizontal(prob=random_flip_horizontal_prop)
        else:
            self.RandomFlipHorizontal = None

        if random_crop_size:
            self.RandomCrop = tlx.vision.transforms.RandomCrop(*random_crop_size)
        else:
            self.RandomCrop = None

    def __call__(self, x, label):
        if self.RandomRotation:
            x = self.RandomRotation(x)

        if self.RandomShift:
            x = self.RandomShift(x)

        if self.RandomFlipHorizontal:
            x = self.RandomFlipHorizontal(x)

        if self.RandomCrop:
            x = self.RandomCrop(x)

        return x, label


class KerasDataRandom(object):
    def __init__(self,
                 random_rotation_degrees=None,
                 random_shift=None,
                 random_flip_horizontal_prop=None,
                 random_crop_size=None,
                 ):
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        self.datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=random_rotation_degrees,  # randomly rotate images in the range (degrees, 0 to 180)
            # randomly shift images horizontally (fraction of total width)
            width_shift_range=random_shift[0] if random_shift else False,
            # randomly shift images vertically (fraction of total height)
            height_shift_range=random_shift[1] if random_shift else False,
            horizontal_flip=True if random_flip_horizontal_prop else False,  # randomly flip images
            vertical_flip=False)  # randomly flip images

    def fit(self, images):
        self.datagen.fit(images)

    def __call__(self, x):
        x = self.datagen.random_transform(x)
        x = self.datagen.standardize(x)

        return x
