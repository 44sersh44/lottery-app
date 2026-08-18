# core/features.py
import numpy as np
from collections import Counter
from typing import List, Tuple

def extract_features(blocks: List[List[int]], total_numbers: int, 
                     lookback: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Преобразует список блоков в матрицу признаков X и бинарные метки y.
    
    Признаки для каждого блока:
    - Частоты чисел за всё время (нормированные)
    - Скользящие средние за последние lookback блоков
    - Позиционные признаки (среднее по позициям)
    - Разности между соседними числами в блоке (средние)
    
    Returns:
        X: (n_blocks, n_features)
        y: (n_blocks, total_numbers) – бинарная матрица: 1 если число присутствует в блоке
    """
    if not blocks:
        return np.array([]), np.array([])
    
    n_blocks = len(blocks)
    pick_count = len(blocks[0])
    
    # ---- Частоты (глобальные) ----
    global_counter = Counter()
    for b in blocks:
        global_counter.update(b)
    total_counts = sum(global_counter.values())
    freq_vec = np.array([global_counter.get(i, 0) / total_counts for i in range(1, total_numbers+1)])
    
    # ---- Скользящие средние за последние lookback блоков ----
    sliding = []
    for i in range(n_blocks):
        window = blocks[max(0, i-lookback):i+1]
        win_counter = Counter()
        for b in window:
            win_counter.update(b)
        win_total = sum(win_counter.values()) or 1
        win_vec = np.array([win_counter.get(j, 0) / win_total for j in range(1, total_numbers+1)])
        sliding.append(win_vec)
    sliding = np.array(sliding)
    
    # ---- Позиционные признаки (среднее каждого числа по позициям) ----
    pos_features = []
    for pos in range(pick_count):
        pos_values = [b[pos] for b in blocks if len(b) > pos]
        if pos_values:
            mean_val = np.mean(pos_values)
            std_val = np.std(pos_values) if len(pos_values) > 1 else 0.0
        else:
            mean_val = total_numbers / 2
            std_val = 0.0
        pos_features.extend([mean_val, std_val])
    pos_vec = np.array(pos_features)
    
    # ---- Разности между соседями ----
    diff_features = []
    for b in blocks:
        if len(b) >= 2:
            diffs = np.diff(sorted(b))
            diff_features.append([np.mean(diffs), np.std(diffs), np.max(diffs), np.min(diffs)])
        else:
            diff_features.append([0, 0, 0, 0])
    diff_features = np.array(diff_features)
    
    # ---- Собираем X ----
    X_list = []
    for i in range(n_blocks):
        # Для каждого блока: глобальные частоты + скользящее среднее в этот момент + позиционные + разности
        row = np.concatenate([
            freq_vec,
            sliding[i],
            pos_vec,
            diff_features[i]
        ])
        X_list.append(row)
    X = np.array(X_list)
    
    # ---- Y: бинарная матрица ----
    y = np.zeros((n_blocks, total_numbers), dtype=np.int8)
    for i, b in enumerate(blocks):
        for num in b:
            if 1 <= num <= total_numbers:
                y[i, num-1] = 1
    
    return X, y