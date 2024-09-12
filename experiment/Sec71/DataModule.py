import os
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
    def __init__(self, normalize=True, append_one=True):
        self.normalize = normalize
        self.append_one = append_one

    def load(self):
        pass

    def fetch(self, n_tr, n_val, n_test, seed=0):
        x, y = self.load()

        # split data
        x_tr, x_val, y_tr, y_val = train_test_split(
            x, y, train_size=n_tr, test_size=n_val + n_test, random_state=seed
        )
        x_val, x_test, y_val, y_test = train_test_split(
            x_val, y_val, train_size=n_val, test_size=n_test, random_state=seed + 1
        )

        # process x
        if self.normalize:
            scaler = StandardScaler()
            scaler.fit(x_tr)
            x_tr = scaler.transform(x_tr)
            x_val = scaler.transform(x_val)
            x_test = scaler.transform(x_test)
        if self.append_one:
            x_tr = np.c_[x_tr, np.ones(n_tr)]
            x_val = np.c_[x_val, np.ones(n_val)]
            x_test = np.c_[x_test, np.ones(n_test)]

        return (x_tr, y_tr), (x_val, y_val), (x_test, y_test)


class MnistModule(DataModule):
    def __init__(self, normalize=True, append_one=False, data_path="data/mnist.npz"):
        import tensorflow as tf

        super().__init__(normalize, append_one)
        self.mnist = tf.keras.datasets.mnist
        self.data_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), data_path
        )

    def load(self):
        (x_train, y_train), (_, _) = self.mnist.load_data(path=self.data_path)

        x_train = x_train.reshape(-1, 28 * 28) / 255.0

        xtr1 = x_train[y_train == 1]
        xtr7 = x_train[y_train == 7]

        x = np.r_[xtr1, xtr7]
        y = np.r_[np.zeros(xtr1.shape[0]), np.ones(xtr7.shape[0])]

        return x, y


class NewsModule(DataModule):
    def __init__(self, normalize=True, append_one=False):
        super().__init__(normalize, append_one)

    def load(self):
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
        vectorizer = TfidfVectorizer(stop_words="english", min_df=0.001, max_df=0.20)
        vectors = vectorizer.fit_transform(newsgroups_train.data)
        vectors_test = vectorizer.transform(newsgroups_test.data)
        x1 = vectors
        y1 = newsgroups_train.target
        x2 = vectors_test
        y2 = newsgroups_test.target
        x = np.array(np.r_[x1.todense(), x2.todense()])
        y = np.r_[y1, y2]
        return x, y


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
