import numpy as np

from datasets.pascal_voc import pascal_voc

__sets = {}

for year in ['2007', '2012']:
    for split in ['train', 'val', 'trainval', 'test']:
        name = f'voc_{year}_{split}'
        __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))

def get_imdb(name):
    """GET an imdb (image database) by name"""
    if name not in __sets:
        raise KeyError(f'Unknown dataset: {name}')
    return __sets[name]()

def list_imdbs():
    return list(__sets.keys())