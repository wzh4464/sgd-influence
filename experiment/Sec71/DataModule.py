###
# File: /DataModule.py
# Created Date: 9th September 2024
# Author: Zihan
# -----
# Last Modified: Thursday, 19th September 2024 9:38:57 pm
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
    def __init__(self, normalize=True, append_one=True, data_dir=None):
        self.normalize = normalize
        self.append_one = append_one
        if data_dir is None:
            self.data_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "data"
            )
        else:
            self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

        # Use the logger configured in the main script
        self.logger = setup_logging(
            "DataModule", 0, output_dir="logs", level=logging.INFO
        )
        self.logger.info(
            f"DataModule initialized: normalize={normalize}, append_one={append_one}, data_dir={self.data_dir}"
        )

    def load(self):
        raise NotImplementedError

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
                self.logger.info(
                    f"Cache file found. Attempting to load from {cache_file}"
                )
                try:
                    with open(cache_file, "rb") as f:
                        result = pickle.load(f)
                    self.logger.info("Successfully loaded data from cache")
                    return result
                except (EOFError, pickle.UnpicklingError) as e:
                    self.logger.error(f"Error loading cache file: {str(e)}")
                    self.logger.info(
                        "Removing corrupted cache file and regenerating data"
                    )
                    os.remove(cache_file)
                except Exception as e:
                    self.logger.error(f"Unexpected error loading cache file: {str(e)}")
                    raise

            self.logger.info(
                "Cache not found or corrupted. Loading and processing data."
            )
            try:
                x, y = self.load()
                self.logger.info(
                    f"Data loaded. Shape of x: {x.shape}, Shape of y: {y.shape}"
                )

                # split data
                x_tr, x_val, y_tr, y_val = train_test_split(
                    x, y, train_size=n_tr, test_size=n_val + n_test, random_state=seed
                )
                x_val, x_test, y_val, y_test = train_test_split(
                    x_val,
                    y_val,
                    train_size=n_val,
                    test_size=n_test,
                    random_state=seed + 1,
                )
                self.logger.info(
                    f"Data split completed. Shapes: x_tr: {x_tr.shape}, x_val: {x_val.shape}, x_test: {x_test.shape}"
                )

                # process x
                if self.normalize:
                    self.logger.info("Normalizing data")
                    scaler = StandardScaler()
                    scaler.fit(x_tr)
                    x_tr = scaler.transform(x_tr)
                    x_val = scaler.transform(x_val)
                    x_test = scaler.transform(x_test)

                if self.append_one:
                    self.logger.info("Appending ones to data")
                    x_tr = np.c_[x_tr, np.ones(n_tr)]
                    x_val = np.c_[x_val, np.ones(n_val)]
                    x_test = np.c_[x_test, np.ones(n_test)]

                result = ((x_tr, y_tr), (x_val, y_val), (x_test, y_test))

                self.logger.info(f"Saving processed data to cache file: {cache_file}")
                with open(cache_file, "wb") as f:
                    pickle.dump(result, f)
                self.logger.info("Data successfully saved to cache")

                return result

            except Exception as e:
                self.logger.error(f"Error in data processing: {str(e)}")
                if os.path.exists(cache_file):
                    os.remove(cache_file)
                raise


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

            x_train = x_train.reshape(-1, 28 * 28) / 255.0

            xtr1 = x_train[y_train == 1]
            xtr7 = x_train[y_train == 7]

            x = np.r_[xtr1, xtr7]
            y = np.r_[np.zeros(xtr1.shape[0]), np.ones(xtr7.shape[0])]

            result = (x, y)

            with open(cache_file, "wb") as f:
                pickle.dump(result, f)

            return result


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
    def __init__(self, normalize=True, append_one=False, csv_path="data"):
        super().__init__(normalize, append_one)
        self.csv_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), csv_path
        )

    def load(self):
        train = pd.read_csv(
            os.path.join(self.csv_path, "adult-training.csv"), names=columns
        )
        test = pd.read_csv(
            os.path.join(self.csv_path, "adult-test.csv"), names=columns, skiprows=1
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
        cache_file = os.path.join(self.data_dir, f"cifar{self.cifar_version}_data.pkl")
        lock_file = cache_file + ".lock"

        with FileLock(lock_file):
            if os.path.exists(cache_file):
                with open(cache_file, "rb") as f:
                    return pickle.load(f)

            if self.cifar_version == 10:
                (x_train, y_train), (x_test, y_test) = (
                    tf.keras.datasets.cifar10.load_data()
                )
            else:  # CIFAR-100
                (x_train, y_train), (x_test, y_test) = (
                    tf.keras.datasets.cifar100.load_data()
                )

            # Combine train and test data
            x = np.vstack((x_train, x_test))
            y = np.vstack((y_train, y_test)).squeeze()

            # Flatten the images
            x = x.reshape(x.shape[0], -1)

            # Normalize pixel values to be between 0 and 1
            x = x.astype("float32") / 255.0

            # For binary classification, we'll use the first two classes
            mask = (y == 0) | (y == 1)
            x = x[mask]
            y = y[mask]

            # Make labels 0 and 1
            y = (y == 1).astype(int)

            # Balanced data sampling: ensure 50/50 class distribution
            class_0_indices = np.where(y == 0)[0]
            class_1_indices = np.where(y == 1)[0]
            min_class_samples = min(len(class_0_indices), len(class_1_indices))

            # Take equal number of samples from both classes
            balanced_indices = np.concatenate(
                [
                    np.random.choice(class_0_indices, min_class_samples, replace=False),
                    np.random.choice(class_1_indices, min_class_samples, replace=False),
                ]
            )

            x = x[balanced_indices]
            y = y[balanced_indices]

            result = (x, y)

            with open(cache_file, "wb") as f:
                pickle.dump(result, f)

            return result


class EMNISTModule:
    def __init__(self, normalize=True, append_one=False, data_dir=None):
        """
        Initialize the EMNISTModule.
        Args:
            normalize (bool): Whether to normalize the data.
            append_one (bool): Whether to append a column of ones to the data.
            data_dir (str): Directory where the data will be stored.
        """
        self.normalize = normalize
        self.append_one = append_one
        if data_dir is None:
            self.data_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "data"
            )
        else:
            self.data_dir = "~/.cache/emnist/"
        os.makedirs(self.data_dir, exist_ok=True)

        # Use the logger configured in the main script
        self.logger = setup_logging(
            "EMNISTModule", 0, output_dir="logs", level=logging.INFO
        )
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

            # Reshape and normalize the images
            x_train = x_train.reshape(-1, 28 * 28).astype("float32") / 255.0

            # Select classes 'A' (label 1) and 'B' (label 2)
            mask = (y_train == 1) | (y_train == 2)
            x_train = x_train[mask]
            y_train = y_train[mask]
            y_train = (y_train == 2).astype(
                int
            )  # Binary classification: 'A' = 0, 'B' = 1

            # Append a column of ones if needed
            if self.append_one:
                x_train = np.c_[x_train, np.ones(x_train.shape[0])]

            # Save to cache
            result = (x_train, y_train)
            with open(cache_file, "wb") as f:
                pickle.dump(result, f)

            self.logger.info("Data saved to cache.")
            return result

    def fetch(self, n_tr, n_val, n_test, seed=0):
        """
        Fetches and splits the data into training, validation, and test sets.

        Args:
            n_tr (int): Number of training samples.
            n_val (int): Number of validation samples.
            n_test (int): Number of test samples.
            seed (int): Random seed for reproducibility.

        Returns:
            Tuple: Split dataset (training, validation, test).
        """
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
                self.logger.info(f"Loading split data from cache file {cache_file}.")
                with open(cache_file, "rb") as f:
                    return pickle.load(f)

            self.logger.info("Cache not found. Loading and splitting data.")
            x, y = self.load()

            # Split data
            from sklearn.model_selection import train_test_split

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

            result = ((x_tr, y_tr), (x_val, y_val), (x_test, y_test))

            # Save the result to cache
            with open(cache_file, "wb") as f:
                pickle.dump(result, f)

            self.logger.info("Data split and saved to cache.")
            return result
