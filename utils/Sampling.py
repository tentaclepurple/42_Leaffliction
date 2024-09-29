import pickle
import os
import random


def get_samp_stats(dic):
    max_class = max(dic, key=dic.get)
    min_class = min(dic, key=dic.get)
    
    max_count = dic[max_class]
    min_count = dic[min_class]
    
    return max_class, max_count, min_class, min_count


def sampling(leaf, method):
    file = leaf + '.pkl'
    with open(file, 'rb') as f:
        dic = pickle.load(f)

    max_class, max_count, min_class, min_count = get_samp_stats(dic)
    print(max_class, max_count, min_class, min_count)

    if method == "oversample":
        del dic[max_class]
        for folder, count in dic.items():
            diff = max_count - count
            for _, _, files in os.walk(f"{leaf}/{folder}"):
                for i in range(diff):
                    print(f"{i}: {random.choice(files)}")

    elif method == "undersample":
        del dic[min_class]
        for folder, count in dic.items():
            diff = count - min_count
            for _, _, files in os.walk(f"{leaf}/{folder}"):
                for i in range(diff):
                    print(f"{folder}: {diff} - > {i}: {random.choice(files)}")
