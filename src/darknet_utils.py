import sys, os
from pathlib import Path

def read_labels(path):
    path = Path(path)
    path = path.parent / (path.stem + '.txt')
    labels = []
    with open(str(path), 'r') as fp:
        lines = fp.readlines()
        for line in lines:
            tokens = line.split(' ')
            if len(tokens) == 5:
                label = (int(tokens[0]), float(tokens[1]), float(tokens[2]), float(tokens[3]), float(tokens[4]))
                labels.append(label)
    return labels

def write_labels(path, labels):
    if len(labels):
        with open(path, 'w') as fp:
            for char_label in labels:
                class_idx, _, cx, cy, w, h = char_label
                fp.write(f'{class_idx} {cx} {cy} {w} {h}\n')

