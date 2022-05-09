import os
import sys

f = open("log.out", 'w')
sys.stdout = f

import tensorflow as tf



print("\nNum GPUs Available:", len(tf.config.experimental.list_physical_devices("GPU")))

f.close()