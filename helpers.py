import pandas
from tqdm.auto import tqdm
import numpy
from typing import Callable, Dict, Any, List


def split_collumn(
    df: pandas.DataFrame,
    column: str,
    new_columns: List[str],
    func: Callable[[Any], List[Any]],
    drop=True,
) -> pandas.DataFrame:
    new_columns_map: Dict[str, List[Any]] = {}
    for new_column in new_columns:
        new_columns_map[new_column] = []
    for val in tqdm(df[column].values):
        splied_val_list = func(val)
        i = 0
        while i < len(new_columns):
            if i < len(splied_val_list):
                new_columns_map[new_columns[i]].append(splied_val_list[i])
            else:
                new_columns_map[new_columns[i]].append(numpy.NaN)
            i += 1
    if drop:
        df = df.drop(columns=[column])
    for name, new_column in new_columns_map.items():
        df[name] = new_column
    return df


def concat_collumn(
    df: pandas.DataFrame,
    columns: List[str],
    new_column: str,
    func: Callable[[Dict[str, Any]], Any],
    drop=True,
) -> pandas.DataFrame:
    new_column_values = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        new_column_values.append(func(row[columns]))
    if drop:
        df = df.drop(columns=columns)
    df[new_column] = new_column_values
    return df
