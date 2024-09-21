###
# File: /DataModule.py
# Created Date: 9th September 2024
# Author: Zihan
# -----
# Last Modified: Saturday, 21st September 2024 9:24:59 pm
# Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
# -----
# HISTORY:
# Date      		By   	Comments
# ----------		------	---------------------------------------------------------
###

import os
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from emnist import extract_training_samples
import tensorflow as tf
import pickle
from filelock import FileLock
import logging
from logging_utils import setup_logging

columns = [
    "Age",
    "Workclass",
    "fnlgwt",
    "Education",
    "Education num",
    "Marital Status",
    "Occupation",
    "Relationship",
    "Race",
    "Sex",
    "Capital Gain",
    "Capital Loss",
    "Hours/Week",
    "Native country",
    "Income",
]


def primary(x):
    if x in [" 1st-4th", " 5th-6th", " 7th-8th", " 9th", " 10th", " 11th", " 12th"]:
        return " Primary"
    else:
        return x


def native(country):
    if country in [" United-States", " Cuba", " 0"]:
        return "US"
    elif country in [
        " England",
        " Germany",
        " Canada",
        " Italy",
        " France",
        " Greece",
        " Philippines",
    ]:
        return "Western"
    elif country in [
        " Mexico",
        " Puerto-Rico",
        " Honduras",
        " Jamaica",
        " Columbia",
        " Laos",
        " Portugal",
        " Haiti",
        " Dominican-Republic",
        " El-Salvador",
        " Guatemala",
        " Peru",
        " Trinadad&Tobago",
        " Outlying-US(Guam-USVI-etc)",
        " Nicaragua",
        " Vietnam",
        " Holand-Netherlands",
    ]:
        return "Poor"  # no offence
    elif country in [
        " India",
        " Iran",
        " Cambodia",
        " Taiwan",
        " Japan",
        " Yugoslavia",
        " China",
        " Hong",
    ]:
        return "Eastern"
    elif country in [
        " South",
        " Poland",
        " Ireland",
        " Hungary",
        " Scotland",
        " Thailand",
        " Ecuador",
    ]:
        return "Poland team"
    else:
        return country


class DataModule:
    def __init__(self, normalize=True, append_one=False, data_dir=None):
        self.normalize = normalize
        self.append_one = append_one
        if data_dir is None:
            self.data_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "data"
            )
        else:
            self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

        self.logger = setup_logging(
            "DataModule", 0, output_dir="logs", level=logging.INFO
        )
        self.logger.info(
            f"DataModule initialized: normalize={normalize}, append_one={append_one}, data_dir={self.data_dir}"
        )

    def load(self):
        raise NotImplementedError

    def preprocess(self, x, y):
        if self.normalize:
            self.logger.info("Normalizing data")
            if x.ndim > 2:  # For image data, normalize per channel
                for i in range(x.shape[-1]):
                    x[..., i] = (x[..., i] - x[..., i].mean()) / x[..., i].std()
            else:
                scaler = StandardScaler()
                x = scaler.fit_transform(x)

        if self.append_one and x.ndim == 2:
            self.logger.info("Appending ones to data")
            x = np.c_[x, np.ones(x.shape[0])]

        return x, y

    def fetch(self, n_tr, n_val, n_test, seed=0):
        cache_file = os.path.join(
            self.data_dir,
            f"{self.__class__.__name__}_{n_tr}_{n_val}_{n_test}_{seed}.pkl",
        )
        lock_file = cache_file + ".lock"
        self.logger.info(
            f"Fetching data with parameters: n_tr={n_tr}, n_val={n_val}, n_test={n_test}, seed={seed}"
        )
        self.logger.info(f"Cache file: {cache_file}")

        with FileLock(lock_file):
            if os.path.exists(cache_file):
                self.logger.info(f"Loading data from cache file {cache_file}")
                with open(cache_file, "rb") as f:
                    return pickle.load(f)

            self.logger.info("Cache not found. Loading and processing data.")
            x, y = self.load()
            self.logger.info(
                f"Data loaded. Shape of x: {x.shape}, Shape of y: {y.shape}"
            )

            x_tr, x_temp, y_tr, y_temp = train_test_split(
                x, y, train_size=n_tr, test_size=n_val + n_test, random_state=seed
            )
            x_val, x_test, y_val, y_test = train_test_split(
                x_temp,
                y_temp,
                train_size=n_val,
                test_size=n_test,
                random_state=seed + 1,
            )

            x_tr, y_tr = self.preprocess(x_tr, y_tr)
            x_val, y_val = self.preprocess(x_val, y_val)
            x_test, y_test = self.preprocess(x_test, y_test)

            result = ((x_tr, y_tr), (x_val, y_val), (x_test, y_test))

            with open(cache_file, "wb") as f:
                pickle.dump(result, f)

            self.logger.info("Data processed and saved to cache.")
            return result


class MnistModule(DataModule):
    def __init__(self, normalize=True, append_one=False, data_dir=None):
        super().__init__(normalize, append_one, data_dir)
        self.mnist = tf.keras.datasets.mnist

    def load(self):
        cache_file = os.path.join(self.data_dir, "mnist_processed_data.pkl")
        lock_file = cache_file + ".lock"

        with FileLock(lock_file):
            if os.path.exists(cache_file):
                with open(cache_file, "rb") as f:
                    return pickle.load(f)

            # Use TensorFlow to load MNIST data
            (x_train, y_train), (_, _) = self.mnist.load_data()

            # Reshape to (samples, channels, height, width)
            x_train = x_train.reshape(-1, 1, 28, 28).astype("float32") / 255.0

            xtr1 = x_train[y_train == 1]
            xtr7 = x_train[y_train == 7]

            x = np.r_[xtr1, xtr7]
            y = np.r_[np.zeros(xtr1.shape[0]), np.ones(xtr7.shape[0])]

            result = (x, y)

            with open(cache_file, "wb") as f:
                pickle.dump(result, f)

            return result

    def preprocess(self, x, y):
        if self.normalize:
            self.logger.info("Normalizing data")
            # Normalize each channel
            x = (x - x.mean(axis=(2, 3), keepdims=True)) / x.std(
                axis=(2, 3), keepdims=True
            )

        return x, y


class NewsModule(DataModule):
    def __init__(self, normalize=True, append_one=False, data_dir=None):
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        super().__init__(normalize, append_one, data_dir)

    def load(self):
        cache_file = os.path.join(self.data_dir, "news_data.pkl")
        lock_file = cache_file + ".lock"

        with FileLock(lock_file):
            if os.path.exists(cache_file):
                with open(cache_file, "rb") as f:
                    return pickle.load(f)

            categories = ["comp.sys.ibm.pc.hardware", "comp.sys.mac.hardware"]
            newsgroups_train = fetch_20newsgroups(
                subset="train",
                remove=("headers", "footers", "quotes"),
                categories=categories,
            )
            newsgroups_test = fetch_20newsgroups(
                subset="test",
                remove=("headers", "footers", "quotes"),
                categories=categories,
            )
            vectorizer = TfidfVectorizer(
                stop_words="english", min_df=0.001, max_df=0.20
            )
            vectors = vectorizer.fit_transform(newsgroups_train.data)
            vectors_test = vectorizer.transform(newsgroups_test.data)
            x1 = vectors
            y1 = newsgroups_train.target
            x2 = vectors_test
            y2 = newsgroups_test.target
            x = np.array(np.r_[x1.todense(), x2.todense()])
            y = np.r_[y1, y2]

            result = (x, y)

            with open(cache_file, "wb") as f:
                pickle.dump(result, f)

            return result


class AdultModule(DataModule):
    def __init__(self, normalize=True, append_one=False, data_dir="data"):
        super().__init__(normalize, append_one)
        self.data_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), data_dir
        )

    def load(self):
        train = pd.read_csv(
            os.path.join(self.data_dir, "adult-training.csv"), names=columns
        )
        test = pd.read_csv(
            os.path.join(self.data_dir, "adult-test.csv"), names=columns, skiprows=1
        )
        df = pd.concat([train, test], ignore_index=True)

        # preprocess
        df.replace(" ?", np.nan, inplace=True)
        df["Income"] = df["Income"].apply(
            lambda x: 1 if x in (" >50K", " >50K.") else 0
        )
        df["Workclass"] = df["Workclass"].fillna(" 0")
        df["Workclass"] = df["Workclass"].replace(" Never-worked", " 0")
        df["fnlgwt"] = df["fnlgwt"].apply(lambda x: np.log1p(x))
        df["Education"] = df["Education"].apply(primary)
        df["Marital Status"] = df["Marital Status"].replace(
            " Married-AF-spouse", " Married-civ-spouse"
        )
        df["Occupation"] = df["Occupation"].fillna(" 0")
        df["Occupation"] = df["Occupation"].replace(" Armed-Forces", " 0")
        df["Native country"] = df["Native country"].fillna(" 0")
        df["Native country"] = df["Native country"].apply(native)

        # one-hot encoding
        categorical_features = df.select_dtypes(include=["object"]).axes[1]
        for col in categorical_features:
            df = pd.concat(
                [df, pd.get_dummies(df[col], prefix=col, prefix_sep=":")], axis=1
            )
            df.drop(col, axis=1, inplace=True)

        # data
        x = df.drop(["Income"], axis=1).values
        y = df["Income"].values
        return x, y


class CifarModule(DataModule):
    def __init__(
        self, cifar_version=10, normalize=True, append_one=False, data_dir=None
    ):
        super().__init__(normalize, append_one, data_dir)
        self.cifar_version = cifar_version

    def load(self):
        if self.cifar_version == 10:
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        else:
            (x_train, y_train), (x_test, y_test) = (
                tf.keras.datasets.cifar100.load_data()
            )

        x = np.vstack((x_train, x_test))
        y = np.vstack((y_train, y_test)).squeeze()

        x = x.astype("float32") / 255.0

        mask = (y == 0) | (y == 1)
        x = x[mask]
        y = y[mask]
        y = (y == 1).astype(int)

        return x, y


class EMNISTModule(DataModule):
    def __init__(self, normalize=True, append_one=False, data_dir=None):
        """
        Initialize the EMNISTModule.
        Args:
            normalize (bool): Whether to normalize the data.
            append_one (bool): Whether to append a column of ones to the data.
            data_dir (str): Directory where the data will be stored.
        """
        if data_dir is None:
            data_dir = os.path.expanduser("/home/zihan/.cache/emnist/")
        super().__init__(normalize, append_one, data_dir)

        self.logger.info(
            f"EMNISTModule initialized: normalize={normalize}, append_one={append_one}, data_dir={self.data_dir}"
        )

    def load(self):
        """
        Loads the EMNIST dataset, processes it, and returns it.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Processed EMNIST dataset (features and labels).
        """
        cache_file = os.path.join(self.data_dir, "emnist_processed_data.pkl")
        lock_file = cache_file + ".lock"

        with FileLock(lock_file):
            if os.path.exists(cache_file):
                self.logger.info("Loading data from cache.")
                with open(cache_file, "rb") as f:
                    return pickle.load(f)

            self.logger.info("Cache not found. Loading EMNIST dataset.")

            # Load 'letters' dataset for binary classification between 'A' and 'B'
            x_train, y_train = extract_training_samples("letters")

            # Reshape the images to (channels, height, width) format
            x_train = x_train.reshape(-1, 1, 28, 28).astype("float32") / 255.0

            # Select classes 'A' (label 1) and 'B' (label 2)
            mask = (y_train == 1) | (y_train == 2)
            x_train = x_train[mask]
            y_train = y_train[mask]
            y_train = (y_train == 2).astype(
                int
            )  # Binary classification: 'A' = 0, 'B' = 1

            # Save to cache
            result = (x_train, y_train)
            with open(cache_file, "wb") as f:
                pickle.dump(result, f)

            self.logger.info("Data saved to cache.")
            return result

    def preprocess(self, x, y):
        """
        Preprocess the data.
        """
        if self.normalize:
            self.logger.info("Normalizing data")
            # For image data, we typically normalize per channel
            for i in range(x.shape[1]):  # Iterate over channels
                x[:, i] = (x[:, i] - x[:, i].mean()) / x[:, i].std()

        # We don't append ones for image data, even if append_one is True
        return x, y


# Registry dictionary to map dataset keys to their corresponding modules
DATA_MODULE_REGISTRY = {}


def register_data_module(key: str, module_class):
    """Register a new data module."""
    if key in DATA_MODULE_REGISTRY:
        raise ValueError(f"Key {key} is already registered.")
    DATA_MODULE_REGISTRY[key] = module_class


# Automatically register existing modules
register_data_module("mnist", MnistModule)
register_data_module("20news", NewsModule)
register_data_module("adult", AdultModule)
register_data_module("cifar", CifarModule)
register_data_module("emnist", EMNISTModule)


def fetch_data_module(key: str, **kwargs):
    """Retrieve a data module class based on the key."""
    if key not in DATA_MODULE_REGISTRY:
        raise ValueError(f"Dataset key {key} is not registered.")
    return DATA_MODULE_REGISTRY[key](**kwargs)
