import argparse
import os
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="Directory that contains the data")

    args = parser.parse_args()
    data_dir = args.data_dir

    # Create GloVe binary file
    glove_file = os.path.join(args.data_dir, 'original/Glove/glove.840B.300d.txt')
    tmp_file = get_tmpfile("glove.840B.300d.w2v.txt")
    _ = glove2word2vec(glove_file, tmp_file)
    model = KeyedVectors.load_word2vec_format(tmp_file)
    model.wv.save_word2vec_format(os.path.join(args.data_dir, "original/Glove/glove.840B.300d.bin"), binary=True)
