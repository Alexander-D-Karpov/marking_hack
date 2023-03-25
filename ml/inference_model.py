import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import time
from datetime import datetime, timedelta
import catboost
import pickle
from sklearn.metrics import mean_absolute_error
from sklearn.base import BaseEstimator, TransformerMixin


# Creating sales lag features
def create_sales_lag_feats(df, gpby_cols, target_col, lags):
    gpby = df.groupby(gpby_cols)
    for i in lags:
        df["_".join([target_col, "lag", str(i)])] = gpby[target_col].shift(
            i
        ).values + np.random.normal(scale=1, size=(len(df),))
    return df


# Creating sales rolling mean features
def create_sales_rmean_feats(
    df, gpby_cols, target_col, windows, min_periods=2, shift=1, win_type=None
):
    gpby = df.groupby(gpby_cols)
    for w in windows:
        df["_".join([target_col, "rmean", str(w)])] = gpby[target_col].shift(
            shift
        ).rolling(
            window=w, min_periods=min_periods, win_type=win_type
        ).mean().values + np.random.normal(
            scale=1, size=(len(df),)
        )
    return df


# Creating sales rolling median features
def create_sales_rmed_feats(
    df, gpby_cols, target_col, windows, min_periods=2, shift=1, win_type=None
):
    gpby = df.groupby(gpby_cols)
    for w in windows:
        df["_".join([target_col, "rmed", str(w)])] = gpby[target_col].shift(
            shift
        ).rolling(
            window=w, min_periods=min_periods, win_type=win_type
        ).median().values + np.random.normal(
            scale=1, size=(len(df),)
        )
    return df


# Creating sales exponentially weighted mean features
def create_sales_ewm_feats(df, gpby_cols, target_col, alpha=[0.9], shift=[1]):
    gpby = df.groupby(gpby_cols)
    for a in alpha:
        for s in shift:
            df["_".join([target_col, "lag", str(s), "ewm", str(a)])] = (
                gpby[target_col].shift(s).ewm(alpha=a).mean().values
            )
    return df


class Preprocesser(BaseEstimator, TransformerMixin):
    def __init__(
        self,
    ):
        self.hash_to_numbers = {}
        self.drop_cols = ["date", "sales", "year"]

    def fit(self, data):
        data = data.rename(
            columns={"dt": "date", "gtin": "item", "id_sp_": "store", "cnt": "sales"}
        )
        data = data.drop(columns="inn", axis=1)
        data = data.dropna(subset="store")
        for col in ["item", "store", "prid"]:
            self.hash_to_numbers[col] = {
                a: b
                for a, b in zip(np.unique(data[col]), np.arange(data[col].nunique()))
            }

        return self

    def get_df(self, data):
        data = data.sort_values(by="dt").reset_index(drop=True)
        data = data.rename(
            columns={"dt": "date", "gtin": "item", "id_sp_": "store", "cnt": "sales"}
        )
        data = data.drop(columns="inn", axis=1)
        data = data.dropna(subset="store")

        for col in ["item", "store", "prid"]:
            data.loc[:, col] = data[col].apply(
                lambda x: self.hash_to_numbers[col][x]
                if x in self.hash_to_numbers[col]
                else np.nan
            )
            data.loc[:, col] = data[col].astype("int")

        data["date"] = data["date"].apply(lambda x: x[:-2] + "01")
        df = (
            data.groupby(["store", "item", "date"])
            .agg(sales=("sales", "sum"), price=("price", "mean"))
            .reset_index()
        )
        return df

    def add_zero_points(self, df):
        all_dates = df["date"].unique()
        x = df.groupby(["store", "item"])["date"].unique()
        add = []
        for store_item, now_dates in tqdm(list(x.items())):
            for d in all_dates:
                if d not in now_dates:
                    add.append(
                        {
                            "store": store_item[0],
                            "item": store_item[1],
                            "date": d,
                            "sales": 0,
                        }
                    )

        df = pd.concat([df, pd.DataFrame(add)])
        return df

    def fill_price(self, prices):
        L = np.ones(len(prices)) * -1
        R = np.ones(len(prices)) * -1
        for i in range(len(prices)):
            if prices[i] == prices[i]:  # not is nan
                L[i] = prices[i]
            elif i > 0:
                L[i] = L[i - 1]

        for i in range(len(prices) - 1, -1, -1):
            if prices[i] == prices[i]:  # not is nan
                R[i] = prices[i]
            elif i != len(prices) - 1:
                R[i] = R[i + 1]

        for i in range(len(prices)):
            if prices[i] != prices[i]:
                if L[i] == -1:
                    prices[i] = R[i]
                elif R[i] == -1:
                    prices[i] = L[i]
                else:
                    prices[i] = (L[i] + R[i]) / 2
        return prices

    def fix_prices(self, df):
        groups = df.sort_values(by="date").groupby(["store", "item"])["price"]

        res = []
        for group in tqdm(groups):
            res += self.fill_price(group[1].values).tolist()
        df.sort_values(by=["store", "item", "date"], axis=0, inplace=True)
        df["price"] = res
        return df

    def build_features(self, df):
        df["date"] = pd.to_datetime(df["date"])
        df["month"] = df.date.dt.month
        df["year"] = df.date.dt.year

        df = create_sales_lag_feats(
            df, gpby_cols=["store", "item"], target_col="sales", lags=[1, 3, 6, 12]
        )

        df = create_sales_rmean_feats(
            df,
            gpby_cols=["store", "item"],
            target_col="sales",
            windows=[2, 3, 6, 12],
            min_periods=2,
            win_type="triang",
        )

        df = create_sales_rmed_feats(
            df,
            gpby_cols=["store", "item"],
            target_col="sales",
            windows=[2, 3, 6, 12],
            min_periods=2,
            win_type=None,
        )

        df = create_sales_ewm_feats(
            df,
            gpby_cols=["store", "item"],
            target_col="sales",
            alpha=[0.95, 0.9, 0.8, 0.7, 0.6, 0.5],
            shift=[1, 3, 6, 12],
        )

        return df.drop(columns=self.drop_cols)

    def transform(self, data, return_y=False):
        df = self.get_df(data)
        df = self.add_zero_points(df)
        df = self.fix_prices(df)

        if return_y:
            y_true = df["sales"]

        df = self.build_features(df)

        if return_y:
            return df, y_true
        else:
            return df


print("loading preprocesser")
with open("ml/preprocesser.pickle", "rb") as f:
    preprocesser = pickle.load(f)

print("loading model")
with open("ml/model.pickle", "rb") as f:
    cat_model = pickle.load(f)


def process_data(data):
    x = preprocesser.transform(data, False)
    preds = cat_model.predict(x)
    return preds
