from __future__ import annotations
from typing import Tuple, Text

import pandas as pd
import torch

from sklearn.preprocessing import StandardScaler, OrdinalEncoder
import numpy as np


def split_okved(code_okved: Text) -> Tuple[Text, Text, Text, Text, Text]:
    """
    Разбивает код ОКВЭД на класс, подкласс, группу, подгруппу, тип
    """
    class_, subclass, group, subgroup, type_ = ['None'] * 5
    assert len(code_okved) in {2, 4, 5, 7, 8}  # Проверяем длину кода
    class_ = code_okved[:2]

    if len(code_okved) >= 4:
        subclass = code_okved[:4]

    if len(code_okved) >= 5:
        group = code_okved[:5]

    if len(code_okved) >= 7:
        subgroup = code_okved[:7]

    if len(code_okved) == 8:
        type_ = code_okved

    return class_, subclass, group, subgroup, type_


def build_nfeat_from_okved_data(okved_data: pd.DataFrame) -> torch.Tensor:
    """
    Создаем из датасета кодов ОКВЭД с обязательными столбцами native_code - исходный код оквэд
    и section_id - раздел кода датасет размеров (n + 1, 6) со столбцами
    класс, подкласс, группу, подгруппу, тип и раздел. Все кодируется в целые числа, а потом нормируется
    """
    # Создаем массив numpy, размером длины кодов ОКВЭД с первой строкой, заполненной
    # 'None'. Значения в первых пяти столбцах данные из кода оквэд после split_okved
    # Данные в последнем столбце section_id - раздел ОКВЭД
    nfeat = np.full((len(okved_data) + 1, 6), 'None', dtype=object)
    nfeat[1:, :-1] = np.array(okved_data['native_code'].map(split_okved).tolist())
    nfeat[1:, -1] = okved_data['section_id'].values

    # Кодируем в просто целые числа
    nfeat = OrdinalEncoder().fit_transform(nfeat).values
    nfeat = StandardScaler().fit_transform(nfeat)  # Нормализуем
    nfeat = torch.from_numpy(nfeat).float()
    return nfeat


def fix_okved(o: str) -> str | None:
    """

    :param o:
    :return:
    """
    if o is None:
        return

    while o[-1] == '0' and '.' in o:
        if o[-2] == '.':
            o = o[:-2]
        else:
            o = o[:-1]
    return o
