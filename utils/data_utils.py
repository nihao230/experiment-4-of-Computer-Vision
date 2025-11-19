import numpy as np
import os
import pickle
import platform

def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(ROOT, "data_batch_%d" % (b,))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, "test_batch"))
    return Xtr, Ytr, Xte, Yte


def get_CIFAR10_data(
    num_training=49000, num_validation=1000, num_test=1000, subtract_mean=True
):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for classifiers. These are the same steps as we used for the SVM, but
    condensed to a single function.
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = "datasets/cifar-10-batches-py"
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    if subtract_mean:
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_val -= mean_image
        X_test -= mean_image

    # Transpose so that channels come first
    X_train = X_train.transpose(0, 3, 1, 2).copy()
    X_val = X_val.transpose(0, 3, 1, 2).copy()
    X_test = X_test.transpose(0, 3, 1, 2).copy()

    # Package data into a dictionary
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }

def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == "2":
        return pickle.load(f)
    elif version[0] == "3":
        return pickle.load(f, encoding="latin1")
    raise ValueError("invalid python version: {}".format(version))


def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, "rb") as f:
        datadict = load_pickle(f)
        X = datadict["data"]
        Y = datadict["labels"]
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y


def load_mnist_idx(filename):
    """
    Load MNIST data from IDX format files.

    Args:
        filename: path to the IDX file

    Returns:
        data: numpy array of the data
    """
    import struct

    with open(filename, 'rb') as f:
        # Read magic number and dimensions
        magic = struct.unpack('>I', f.read(4))[0]

        if magic == 2051:  # Image file
            num_images = struct.unpack('>I', f.read(4))[0]
            num_rows = struct.unpack('>I', f.read(4))[0]
            num_cols = struct.unpack('>I', f.read(4))[0]

            # Read all image data
            data = np.frombuffer(f.read(), dtype=np.uint8)
            data = data.reshape(num_images, num_rows, num_cols)

        elif magic == 2049:  # Label file
            num_labels = struct.unpack('>I', f.read(4))[0]

            # Read all label data
            data = np.frombuffer(f.read(), dtype=np.uint8)

        else:
            raise ValueError(f"Invalid magic number {magic} in MNIST file")

    return data


def get_MNIST_data(
    num_training=55000, num_validation=5000, num_test=10000, subtract_mean=True
):
    """
    Load the MNIST dataset from local files and perform preprocessing to prepare
    it for classifiers.

    Returns a dictionary with keys:
        X_train: training images, shape (num_training, 1, 28, 28)
        y_train: training labels, shape (num_training,)
        X_val: validation images, shape (num_validation, 1, 28, 28)
        y_val: validation labels, shape (num_validation,)
        X_test: test images, shape (num_test, 1, 28, 28)
        y_test: test labels, shape (num_test,)
    """
    # Load MNIST data from local directory
    mnist_dir = "datasets/mnist"

    # Load training data
    X_train_full = load_mnist_idx(os.path.join(mnist_dir, "train-images.idx3-ubyte"))
    y_train_full = load_mnist_idx(os.path.join(mnist_dir, "train-labels.idx1-ubyte"))

    # Load test data
    X_test = load_mnist_idx(os.path.join(mnist_dir, "t10k-images.idx3-ubyte"))
    y_test = load_mnist_idx(os.path.join(mnist_dir, "t10k-labels.idx1-ubyte"))

    # Convert to float
    X_train_full = X_train_full.astype('float')
    X_test = X_test.astype('float')

    # Subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train_full[mask]
    y_val = y_train_full[mask]
    mask = list(range(num_training))
    X_train = X_train_full[mask]
    y_train = y_train_full[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    if subtract_mean:
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_val -= mean_image
        X_test -= mean_image

    # Add channel dimension and ensure shape is (N, 1, 28, 28)
    X_train = X_train[:, np.newaxis, :, :].copy()
    X_val = X_val[:, np.newaxis, :, :].copy()
    X_test = X_test[:, np.newaxis, :, :].copy()

    # Package data into a dictionary
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }

