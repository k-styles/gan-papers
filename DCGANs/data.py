import tensorflow.datasets as tfds
import tensorflow as tf

data = input("What data you want to train on? (mnist, LSUN): ")
if data == "mnist":
    #TODO: Load LSUN Dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
else:
    logging.error(f"Not yet implemented for {data}. This should be either reported or feel free to make a pull request.")

fig = plt.figure(figsize=(10, 7))

do_you = input("Do you want to visualize data?: (yes, no): ")

if(do_you == "yes"):
    rows_in = input("Number of rows you want: ")
    columns_in = input("Number of columns you want: ")
    rows = int(rows_in)
    columns = int(columns_in)
    for i in range(rows * columns):
        fig.add_subplot(rows, columns, i + 1)
        plt.imshow(x_train[i], cmap='gray')
        plt.axis('off')

    plt.show()
