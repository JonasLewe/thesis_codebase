import tensorflow as tf

class JoinedGenerator2View(tf.keras.utils.Sequence):
    def __init__(self, generator_ppl, generator_xpl):
        self.generator_ppl = generator_ppl
        self.generator_xpl = generator_xpl 
        self.classes = generator_ppl.classes
        self.samples = generator_ppl.samples
        self.batch_size = generator_ppl.batch_size
        self.class_indices = generator_ppl.class_indices

    def __len__(self):
        return len(self.generator_xpl)

    def __getitem__(self, i):
        x1, y1 = self.generator_ppl[i]
        x2, y2 = self.generator_xpl[i]
        return [x1, x2], [y1, y2]

    def on_epoch_end(self):
        self.generator_ppl.on_epoch_end()
        self.generator_xpl.on_epoch_end()


class JoinedGenerator3View(tf.keras.utils.Sequence):
    # use 3 different generators for 3 xpl views
    def __init__(self, generator_xpl0, generator_xpl30, generator_xpl60):
        self.generator_xpl0 = generator_xpl0
        self.generator_xpl30 = generator_xpl30
        self.generator_xpl60 = generator_xpl60

        self.classes = generator_xpl0.classes
        self.samples = generator_xpl0.samples
        self.batch_size = generator_xpl0.batch_size
        self.class_indices = generator_xpl0.class_indices

    def __len__(self):
        return len(self.generator_xpl0)

    def __getitem__(self, i):
        x1, y1 = self.generator_xpl0[i]
        x2, y2 = self.generator_xpl30[i]
        x3, y3 = self.generator_xpl60[i]
        return [x1, x2, x3], [y1, y2, y3]

    def on_epoch_end(self):        
        self.generator_xpl0.on_epoch_end()
        self.generator_xpl30.on_epoch_end()
        self.generator_xpl60.on_epoch_end()

    
class JoinedGenerator6View(tf.keras.utils.Sequence):
    # use 6 different generators, 3 for ppl 3 for xpl
    def __init__(self,
                 generator_ppl0,
                 generator_ppl30,
                 generator_ppl60,
                 generator_xpl0,
                 generator_xpl30,
                 generator_xpl60
                ):
        self.generator_ppl0 = generator_ppl0
        self.generator_ppl30 = generator_ppl30
        self.generator_ppl60 = generator_ppl60

        self.generator_xpl0 = generator_xpl0
        self.generator_xpl30 = generator_xpl30
        self.generator_xpl60 = generator_xpl60

        self.classes = generator_xpl0.classes
        self.samples = generator_xpl0.samples
        self.batch_size = generator_xpl0.batch_size
        self.class_indices = generator_xpl0.class_indices

    def __len__(self):
        return len(self.generator_xpl0)

    def __getitem__(self, i):
        x1, y1 = self.generator_ppl0[i]
        x2, y2 = self.generator_ppl30[i]
        x3, y3 = self.generator_ppl60[i]

        x4, y4 = self.generator_xpl0[i]
        x5, y5 = self.generator_xpl30[i]
        x6, y6 = self.generator_xpl60[i]
        return [x1, x2, x3, x4, x5, x6], [y1, y2, y3, y4, y5, y6]

    def on_epoch_end(self):
        self.generator_ppl0.on_epoch_end()
        self.generator_ppl30.on_epoch_end()
        self.generator_ppl60.on_epoch_end()
        
        self.generator_xpl0.on_epoch_end()
        self.generator_xpl30.on_epoch_end()
        self.generator_xpl60.on_epoch_end()
    