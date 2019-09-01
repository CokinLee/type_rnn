# -*- coding: utf-8 -*-  
# 模型的加载及使用
from keras.models import load_model
import numpy as np
import sys
import datetime

if len(sys.argv) <= 1:
    print("请输入参数")
    exit(1)

DIGITS = 3
REVERSE = True # 输入倒序，补充的空格留在前面

# Maximum length of input is 'int kg = _g' (e.g., '345+678'). Maximum length of
# int is DIGITS.
# MAXLEN = DIGITS + 5
MAXLEN = 20

class CharacterTable(object):
    """Given a set of characters:
    + Encode them to a one-hot integer representation
    + Decode the one-hot or integer representation to their character output
    + Decode a vector of probabilities to their character output
    """
    def __init__(self, chars):
        """Initialize character table.

        # Arguments
            chars: Characters that can appear in the input.
        """
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

    def encode(self, C, num_rows):
        """One-hot encode given string C.

        # Arguments
            C: string, to be encoded.
            num_rows: Number of rows in the returned one-hot encoding. This is
                used to keep the # of rows for each data the same.
        """
        x = np.zeros((num_rows, len(self.chars)))
        for i, c in enumerate(C):
            x[i, self.char_indices[c]] = 1
        return x

    def decode(self, x, calc_argmax=True):
        """Decode the given vector or 2D array to their character output.

        # Arguments
            x: A vector or a 2D array of probabilities or one-hot representations;
                or a vector of character indices (used with `calc_argmax=False`).
            calc_argmax: Whether to find the character index with maximum
                probability, defaults to `True`.
        """
        if calc_argmax:
            x = x.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in x)


chars = '0123456789%.-+*_/^~|=><()[]:,ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz '
ctable = CharacterTable(chars)

print("Using loaded model to predict...")
load_model = load_model("data/test/exps_rnn_model.h5")
unknown = sys.argv[1]
query = unknown + ' ' * (MAXLEN - len(unknown))
    # Answers can be of maximum size DIGITS + 1.
if REVERSE:
    # Reverse the query, e.g., '12+345  ' becomes '  543+21'. (Note the
    # space used for padding.)
    query = query[::-1]
x = ctable.encode(query, MAXLEN)
imagelist = []
imagelist.append(x)

print("算式: " + unknown)
begin = datetime.datetime.now()
predicted = load_model.predict(np.asarray(imagelist))

# print("\nPredicted softmax vector is: ")
# print(predicted)

guess = ctable.decode(predicted[0])
print("结果: " + guess)
end = datetime.datetime.now()
k = end - begin
print("用时: " + str(k.total_seconds()) +"s")