import tensorflow as tf
from generator_model import generator
from discriminator_model import discriminator
from matplotlib import pyplot as plt
import logging

data = input("What data you want to train on? (mnist, tfd): ")
if data == "mnist":
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
else:
    logging.error(f"Not yet implemented for {data}. This should be either reported or can be contributed.")

fig = plt.figure(figsize=(10, 7))

do_you = input("Do you want to visualize data?: (yes, no)")

if(do_you == "yes"):
    rows_in = input("Number of rows you want: ")
    columns_in = input("Number of columns you want: ")
    rows = int(rows_in)
    columns = int(columns_in)
    for i in range(rows * columns):
        fig.add_subplot(rows, columns, i + 1)
        plt.imshow(x_train[i])
        plt.axis('off')

    plt.show()



tf.random.set_seed(5)
random_tensor = tf.random.normal(shape=[10], mean=0.0, stddev=1.0)

generator = generator.Generator()
generator.build(input_shape=random_tensor.shape)
print(generator.summary())

tf.random.set_seed(5)
random_tensor_image = tf.random.normal(shape=[28,28], mean=0.0, stddev=1.0)

discriminator = discriminator.Discriminator()
discriminator.build(input_shape=random_tensor_image.shape)
print(discriminator.summary())