# utils/helpers.py

# Mapping dari index ke flare class
index_to_class_map = {
    0: "A",
    1: "B",
    2: "C",
    3: "M",
    4: "X"
}

# Bisa ditambahkan juga kebalikannya jika diperlukan
class_to_index_map = {v: k for k, v in index_to_class_map.items()}
