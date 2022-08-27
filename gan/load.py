import tensorflow as tf
from generator_model import generator
from discriminator_model import discriminator
from matplotlib import pyplot as plt
import logging
from training_toolkit import train_tools
import random
tf.config.run_functions_eagerly(False)

loaded_gen = generator.Generator()
loaded_discr = discriminator.Discriminator()

loaded_gen.load_weights("saved_models/generator_trained/generator_trained.index")
loaded_discr.load_weights("saved_models/discriminator_trained/discriminator_trained.index")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
gen_rows = 10
gen_columns = 10
# for i in range(gen_rows * gen_columns):
#     test_noise_sample = tf.random.uniform(shape=[100], minval=0.0, maxval=1.0)
#     plt.imshow(loaded_gen(input=test_noise_sample), cmap="gray")
#     plt.show()
test_noise_sample = tf.random.uniform(shape=[100], minval=0.0, maxval=1.0)
print(loaded_discr(x_train[0]))#loaded_gen(test_noise_sample)))
# for i in range(gen_rows * gen_columns):
#     fig_generated.add_subplot(gen_rows, gen_columns, i + 1)
#     plt.imshow(generator(test_noise_data[i]), cmap="gray")
#     plt.xticks([])
#     plt.yticks([])
plt.imshow(loaded_gen(input=test_noise_sample), cmap="gray")
plt.show()
