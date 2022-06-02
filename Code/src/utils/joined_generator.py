import tensorflow as tf

class JoinedGenerator(tf.keras.utils.Sequence):
    def __init__(self, generator1, generator2):
        self.generator1 = generator1
        self.generator2 = generator2 
        self.classes = generator1.classes
        self.samples = generator1.samples
        self.batch_size = generator1.batch_size
        self.class_indices = generator1.class_indices

    def __len__(self):
        return len(self.generator1)

    def __getitem__(self, i):
        x1, y1 = self.generator1[i]
        x2, y2 = self.generator2[i]
        return [x1, x2], [y1, y2]

    def on_epoch_end(self):
        self.generator1.on_epoch_end()
        self.generator2.on_epoch_end()
    