import tensorflow as tf
from transformers import BertTokenizer

# Define paths
en_file_path = 'path/to/flores101_dataset/eng_Latn.dev'
bn_file_path = 'path/to/flores101_dataset/ben_Beng.dev'

# Load the dataset
with open(en_file_path, 'r', encoding='utf-8') as f:
    en_lines = f.readlines()

with open(bn_file_path, 'r', encoding='utf-8') as f:
    bn_lines = f.readlines()

assert len(en_lines) == len(bn_lines)

# Tokenization
tokenizer_en = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer_bn = BertTokenizer.from_pretrained('l3cube-pune/bengali-bert')

def encode(lang1, lang2):
    lang1 = [tokenizer_en.encode(s.strip()) for s in lang1]
    lang2 = [tokenizer_bn.encode(s.strip()) for s in lang2]
    return lang1, lang2

en_tokens, bn_tokens = encode(en_lines, bn_lines)

# Padding
BUFFER_SIZE = 20000
BATCH_SIZE = 64
MAX_LENGTH = 40  # Adjust based on your dataset

def filter_max_length(x, y, max_length=MAX_LENGTH):
    return tf.logical_and(tf.size(x) <= max_length, tf.size(y) <= max_length)

dataset = tf.data.Dataset.from_tensor_slices((en_tokens, bn_tokens))
dataset = dataset.filter(filter_max_length)
dataset = dataset.cache()
dataset = dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE, padded_shapes=([-1], [-1]))
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)


print(tokenizer_bn)