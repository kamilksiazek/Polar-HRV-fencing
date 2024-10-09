import pandas as pd


def load_arbitrary_dataframe(folder, name):
    """
    Load Pandas dataframe with an arbitrary name convention.

    Arguments:
    ----------
       *folder*: (string) folder with experiment's files
       *name*: (string) name of the file with data

    Returns:
    --------
       *data*: Pandas dataframe with loaded data
    """
    data = pd.read_csv(
        f'{folder}{name}',
        delimiter=';'
    )
    return data


if __name__ == "__main__":
    pass
