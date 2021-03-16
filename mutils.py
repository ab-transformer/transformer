import time
from pathlib import Path

csv_dir = Path("logs/csv")


class ElapsedTime:
    def __init__(
        self, process_name: str = "Process", is_ms: bool = False, verbose: bool = True
    ):
        """
        This class should be used in with statement. It measures the time spend in the statement.
        :param process_name: Name of the process. Used in the displayed string.
        :param is_ms: If true the output is in milliseconds, seconds otherwise.
        :param verbose: If true displays runtime message on the console
        """
        self.verbose = verbose
        self.is_ms = is_ms
        self.process_name = process_name
        self.start = -1
        self.end = -1
        self.elapsed_time_ms = -1

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, ignored, ignored1, ignored2):
        self.end = time.time()
        elapsed_time = self.end - self.start
        self.elapsed_time_ms = (self.end - self.start) * 1000
        if self.verbose:
            if self.is_ms:
                print(
                    "%s was running for %f ms"
                    % (self.process_name, self.elapsed_time_ms)
                )
            else:
                print("%s was running for %f s" % (self.process_name, elapsed_time))


def get_epoch_info(df):
    df_train = df[~df["train_1mae_epoch"].isnull()][["epoch", "train_1mae_epoch"]]
    df_valid = df[~df["valid_1mae"].isnull()][["epoch", "valid_1mae"]]
    df_res = df_train.merge(df_valid, on="epoch")
    if "debug_early_stopping_best_score_epoch" in df.keys():
        df_debug = df[~df["debug_early_stopping_wait_count_epoch"].isnull()][
            [
                "epoch",
                "debug_early_stopping_wait_count_epoch",
                "debug_early_stopping_best_score_epoch",
            ]
        ]
        df_res = df_res.merge(df_debug, on="epoch")
    return df_res
