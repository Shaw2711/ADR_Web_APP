import os
import pandas as pd


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_DIR = os.path.join(BASE_DIR, "backend/data")


class DataLoader:

    def __init__(self):

        self.fp_df = pd.read_csv(
            os.path.join(DATA_DIR,
            "intersection_Fingerprint_mat_new_Data.csv"),
            index_col=0
        )

        self.dgi_df = pd.read_csv(
            os.path.join(DATA_DIR,
            "intersection_DGIdb_mat_new.csv"),
            index_col=0
        )

        self.adr_df = pd.read_csv(
            os.path.join(DATA_DIR,
            "your_ADR_matrix.csv"),
            index_col=0
        )
    def get_all(self):

        return (
            self.dgi_df.values,
            self.fp_df.values,
            self.adr_df.values
        )
    def align(self):

        common = sorted(
            set(self.fp_df.index)
            & set(self.dgi_df.index)
            & set(self.adr_df.index)
        )

        self.fp_df = self.fp_df.loc[common]
        self.dgi_df = self.dgi_df.loc[common]
        self.adr_df = self.adr_df.loc[common]


loader = DataLoader()