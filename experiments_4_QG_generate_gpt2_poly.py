# STEP 4: use trained FQG model to generate new QG data using augmented sentences

import os
import argparse
import logging
from polyaxon_client.tracking import get_outputs_path

# Set up the logger
logging.basicConfig(format='%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, help="Directory containing the data for the ACS-QG project")
    args = parser.parse_args()

    data_dir = args.data_dir.rstrip("/") + "/"
    OUT_DIR = get_outputs_path().rstrip("/") + "/"

    # debug
    processed_path = data_dir + "processed/SQuAD2.0/"
    data_file_prefix = "train"
    st_idx = str(0)
    ed_idx = str(50000)
    os.system('CUDA_VISIBLE_DEVICES=0 PYTHONIOENCODING=utf-8 python3 QG_gpt2_generate.py  \
        --model_type gpt2 \
        --model_name_or_path ' + data_dir + 'output/QG/gpt2_question_generation/4epochs/2batchsize/ \
        --filename "' + processed_path + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.pkl" \
        --filecache "' + OUT_DIR + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.cache.qg.gpt2.pth" \
        --data_type augmented_sents \
        --output_file "' + OUT_DIR + data_file_prefix + '.qa.' + st_idx + '_' + ed_idx + '.qg.generated.gpt2.4e8b.debug.json" \
        --debug')

    # squad data
    processed_path = data_dir + "processed/SQuAD2.0/"
    data_file_prefix = "train"
    st_idx = str(0)
    ed_idx = str(50000)
    os.system('CUDA_VISIBLE_DEVICES=0 PYTHONIOENCODING=utf-8 python3 QG_gpt2_generate.py  \
        --model_type gpt2 \
        --model_name_or_path ' + data_dir + 'output/QG/gpt2_question_generation/4epochs/2batchsize/ \
        --filename "' + processed_path + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.pkl" \
        --filecache "' + OUT_DIR + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.cache.qg.gpt2.pth" \
        --data_type augmented_sents \
        --output_file "' + OUT_DIR + data_file_prefix + '.qa.' + st_idx + '_' + ed_idx + '.qg.generated.gpt2.json"')

    processed_path = data_dir + "processed/SQuAD2.0/"
    data_file_prefix = "train"
    st_idx = str(50000)
    ed_idx = str(92210)
    os.system('CUDA_VISIBLE_DEVICES=0 PYTHONIOENCODING=utf-8 python3 QG_gpt2_generate.py  \
        --model_type gpt2 \
        --model_name_or_path ' + data_dir + 'output/QG/gpt2_question_generation/4epochs/2batchsize/ \
        --filename "' + processed_path + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.pkl" \
        --filecache "' + OUT_DIR + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.cache.qg.gpt2.pth" \
        --data_type augmented_sents \
        --output_file "' + OUT_DIR + data_file_prefix + '.qa.' + st_idx + '_' + ed_idx + '.qg.generated.gpt2.json"')

    # wiki data
    processed_path = data_dir + "processed/Wiki10000/"
    data_file_prefix = "wiki10000"
    st_idx = str(0)
    ed_idx = str(50000)
    os.system('CUDA_VISIBLE_DEVICES=0 PYTHONIOENCODING=utf-8 python3 QG_gpt2_generate.py  \
        --model_type gpt2 \
        --model_name_or_path ' + data_dir + 'output/QG/gpt2_question_generation/4epochs/2batchsize/ \
        --filename "' + processed_path + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.pkl" \
        --filecache "' + OUT_DIR + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.cache.qg.gpt2.pth" \
        --data_type augmented_sents \
        --output_file "' + OUT_DIR + data_file_prefix + '.qa.' + st_idx + '_' + ed_idx + '.qg.generated.gpt2.json"')

    processed_path = data_dir + "processed/Wiki10000/"
    data_file_prefix = "wiki10000"
    st_idx = str(50000)
    ed_idx = str(100000)
    os.system('CUDA_VISIBLE_DEVICES=0 PYTHONIOENCODING=utf-8 python3 QG_gpt2_generate.py  \
        --model_type gpt2 \
        --model_name_or_path ' + data_dir + 'output/QG/gpt2_question_generation/4epochs/2batchsize/ \
        --filename "' + processed_path + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.pkl" \
        --filecache "' + OUT_DIR + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.cache.qg.gpt2.pth" \
        --data_type augmented_sents \
        --output_file "' + OUT_DIR + data_file_prefix + '.qa.' + st_idx + '_' + ed_idx + '.qg.generated.gpt2.json"')

    processed_path = data_dir + "processed/Wiki10000/"
    data_file_prefix = "wiki10000"
    st_idx = str(100000)
    ed_idx = str(150000)
    os.system('CUDA_VISIBLE_DEVICES=0 PYTHONIOENCODING=utf-8 python3 QG_gpt2_generate.py  \
        --model_type gpt2 \
        --model_name_or_path ' + data_dir + 'output/QG/gpt2_question_generation/4epochs/2batchsize/ \
        --filename "' + processed_path + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.pkl" \
        --filecache "' + OUT_DIR + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.cache.qg.gpt2.pth" \
        --data_type augmented_sents \
        --output_file "' + OUT_DIR + data_file_prefix + '.qa.' + st_idx + '_' + ed_idx + '.qg.generated.gpt2.json"')

    processed_path = data_dir + "processed/Wiki10000/"
    data_file_prefix = "wiki10000"
    st_idx = str(150000)
    ed_idx = str(200000)
    os.system('CUDA_VISIBLE_DEVICES=0 PYTHONIOENCODING=utf-8 python3 QG_gpt2_generate.py  \
        --model_type gpt2 \
        --model_name_or_path ' + data_dir + 'output/QG/gpt2_question_generation/4epochs/2batchsize/ \
        --filename "' + processed_path + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.pkl" \
        --filecache "' + OUT_DIR + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.cache.qg.gpt2.pth" \
        --data_type augmented_sents \
        --output_file "' + OUT_DIR + data_file_prefix + '.qa.' + st_idx + '_' + ed_idx + '.qg.generated.gpt2.json"')

    processed_path = data_dir + "processed/Wiki10000/"
    data_file_prefix = "wiki10000"
    st_idx = str(200000)
    ed_idx = str(250000)
    os.system('CUDA_VISIBLE_DEVICES=0 PYTHONIOENCODING=utf-8 python3 QG_gpt2_generate.py  \
        --model_type gpt2 \
        --model_name_or_path ' + data_dir + 'output/QG/gpt2_question_generation/4epochs/2batchsize/ \
        --filename "' + processed_path + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.pkl" \
        --filecache "' + OUT_DIR + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.cache.qg.gpt2.pth" \
        --data_type augmented_sents \
        --output_file "' + OUT_DIR + data_file_prefix + '.qa.' + st_idx + '_' + ed_idx + '.qg.generated.gpt2.json"')

    processed_path = data_dir + "processed/Wiki10000/"
    data_file_prefix = "wiki10000"
    st_idx = str(250000)
    ed_idx = str(300000)
    os.system('CUDA_VISIBLE_DEVICES=0 PYTHONIOENCODING=utf-8 python3 QG_gpt2_generate.py  \
        --model_type gpt2 \
        --model_name_or_path ' + data_dir + 'output/QG/gpt2_question_generation/4epochs/2batchsize/ \
        --filename "' + processed_path + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.pkl" \
        --filecache "' + OUT_DIR + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.cache.qg.gpt2.pth" \
        --data_type augmented_sents \
        --output_file "' + OUT_DIR + data_file_prefix + '.qa.' + st_idx + '_' + ed_idx + '.qg.generated.gpt2.json"')

    processed_path = data_dir + "processed/Wiki10000/"
    data_file_prefix = "wiki10000"
    st_idx = str(300000)
    ed_idx = str(350000)
    os.system('CUDA_VISIBLE_DEVICES=0 PYTHONIOENCODING=utf-8 python3 QG_gpt2_generate.py  \
        --model_type gpt2 \
        --model_name_or_path ' + data_dir + 'output/QG/gpt2_question_generation/4epochs/2batchsize/ \
        --filename "' + processed_path + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.pkl" \
        --filecache "' + OUT_DIR + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.cache.qg.gpt2.pth" \
        --data_type augmented_sents \
        --output_file "' + OUT_DIR + data_file_prefix + '.qa.' + st_idx + '_' + ed_idx + '.qg.generated.gpt2.json"')

    processed_path = data_dir + "processed/Wiki10000/"
    data_file_prefix = "wiki10000"
    st_idx = str(350000)
    ed_idx = str(400000)
    os.system('CUDA_VISIBLE_DEVICES=0 PYTHONIOENCODING=utf-8 python3 QG_gpt2_generate.py  \
        --model_type gpt2 \
        --model_name_or_path ' + data_dir + 'output/QG/gpt2_question_generation/4epochs/2batchsize/ \
        --filename "' + processed_path + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.pkl" \
        --filecache "' + OUT_DIR + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.cache.qg.gpt2.pth" \
        --data_type augmented_sents \
        --output_file "' + OUT_DIR + data_file_prefix + '.qa.' + st_idx + '_' + ed_idx + '.qg.generated.gpt2.json"')

    processed_path = data_dir + "processed/Wiki10000/"
    data_file_prefix = "wiki10000"
    st_idx = str(400000)
    ed_idx = str(450000)
    os.system('CUDA_VISIBLE_DEVICES=0 PYTHONIOENCODING=utf-8 python3 QG_gpt2_generate.py  \
        --model_type gpt2 \
        --model_name_or_path ' + data_dir + 'output/QG/gpt2_question_generation/4epochs/2batchsize/ \
        --filename "' + processed_path + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.pkl" \
        --filecache "' + OUT_DIR + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.cache.qg.gpt2.pth" \
        --data_type augmented_sents \
        --output_file "' + OUT_DIR + data_file_prefix + '.qa.' + st_idx + '_' + ed_idx + '.qg.generated.gpt2.json"')

    processed_path = data_dir + "processed/Wiki10000/"
    data_file_prefix = "wiki10000"
    st_idx = str(450000)
    ed_idx = str(500000)
    os.system('CUDA_VISIBLE_DEVICES=0 PYTHONIOENCODING=utf-8 python3 QG_gpt2_generate.py  \
        --model_type gpt2 \
        --model_name_or_path ' + data_dir + 'output/QG/gpt2_question_generation/4epochs/2batchsize/ \
        --filename "' + processed_path + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.pkl" \
        --filecache "' + OUT_DIR + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.cache.qg.gpt2.pth" \
        --data_type augmented_sents \
        --output_file "' + OUT_DIR + data_file_prefix + '.qa.' + st_idx + '_' + ed_idx + '.qg.generated.gpt2.json"')

    processed_path = data_dir + "processed/Wiki10000/"
    data_file_prefix = "wiki10000"
    st_idx = str(500000)
    ed_idx = str(550000)
    os.system('CUDA_VISIBLE_DEVICES=0 PYTHONIOENCODING=utf-8 python3 QG_gpt2_generate.py  \
        --model_type gpt2 \
        --model_name_or_path ' + data_dir + 'output/QG/gpt2_question_generation/4epochs/2batchsize/ \
        --filename "' + processed_path + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.pkl" \
        --filecache "' + OUT_DIR + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.cache.qg.gpt2.pth" \
        --data_type augmented_sents \
        --output_file "' + OUT_DIR + data_file_prefix + '.qa.' + st_idx + '_' + ed_idx + '.qg.generated.gpt2.json"')

    processed_path = data_dir + "processed/Wiki10000/"
    data_file_prefix = "wiki10000"
    st_idx = str(550000)
    ed_idx = str(600000)
    os.system('CUDA_VISIBLE_DEVICES=0 PYTHONIOENCODING=utf-8 python3 QG_gpt2_generate.py  \
        --model_type gpt2 \
        --model_name_or_path ' + data_dir + 'output/QG/gpt2_question_generation/4epochs/2batchsize/ \
        --filename "' + processed_path + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.pkl" \
        --filecache "' + OUT_DIR + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.cache.qg.gpt2.pth" \
        --data_type augmented_sents \
        --output_file "' + OUT_DIR + data_file_prefix + '.qa.' + st_idx + '_' + ed_idx + '.qg.generated.gpt2.json"')

    processed_path = data_dir + "processed/Wiki10000/"
    data_file_prefix = "wiki10000"
    st_idx = str(600000)
    ed_idx = str(650000)
    os.system('CUDA_VISIBLE_DEVICES=0 PYTHONIOENCODING=utf-8 python3 QG_gpt2_generate.py  \
        --model_type gpt2 \
        --model_name_or_path ' + data_dir + 'output/QG/gpt2_question_generation/4epochs/2batchsize/ \
        --filename "' + processed_path + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.pkl" \
        --filecache "' + OUT_DIR + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.cache.qg.gpt2.pth" \
        --data_type augmented_sents \
        --output_file "' + OUT_DIR + data_file_prefix + '.qa.' + st_idx + '_' + ed_idx + '.qg.generated.gpt2.json"')

    processed_path = data_dir + "processed/Wiki10000/"
    data_file_prefix = "wiki10000"
    st_idx = str(650000)
    ed_idx = str(700000)
    os.system('CUDA_VISIBLE_DEVICES=0 PYTHONIOENCODING=utf-8 python3 QG_gpt2_generate.py  \
        --model_type gpt2 \
        --model_name_or_path ' + data_dir + 'output/QG/gpt2_question_generation/4epochs/2batchsize/ \
        --filename "' + processed_path + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.pkl" \
        --filecache "' + OUT_DIR + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.cache.qg.gpt2.pth" \
        --data_type augmented_sents \
        --output_file "' + OUT_DIR + data_file_prefix + '.qa.' + st_idx + '_' + ed_idx + '.qg.generated.gpt2.json"')

    processed_path = data_dir + "processed/Wiki10000/"
    data_file_prefix = "wiki10000"
    st_idx = str(700000)
    ed_idx = str(750000)
    os.system('CUDA_VISIBLE_DEVICES=0 PYTHONIOENCODING=utf-8 python3 QG_gpt2_generate.py  \
        --model_type gpt2 \
        --model_name_or_path ' + data_dir + 'output/QG/gpt2_question_generation/4epochs/2batchsize/ \
        --filename "' + processed_path + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.pkl" \
        --filecache "' + OUT_DIR + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.cache.qg.gpt2.pth" \
        --data_type augmented_sents \
        --output_file "' + OUT_DIR + data_file_prefix + '.qa.' + st_idx + '_' + ed_idx + '.qg.generated.gpt2.json"')

    processed_path = data_dir + "processed/Wiki10000/"
    data_file_prefix = "wiki10000"
    st_idx = str(750000)
    ed_idx = str(800000)
    os.system('CUDA_VISIBLE_DEVICES=0 PYTHONIOENCODING=utf-8 python3 QG_gpt2_generate.py  \
        --model_type gpt2 \
        --model_name_or_path ' + data_dir + 'output/QG/gpt2_question_generation/4epochs/2batchsize/ \
        --filename "' + processed_path + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.pkl" \
        --filecache "' + OUT_DIR + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.cache.qg.gpt2.pth" \
        --data_type augmented_sents \
        --output_file "' + OUT_DIR + data_file_prefix + '.qa.' + st_idx + '_' + ed_idx + '.qg.generated.gpt2.json"')

    processed_path = data_dir + "processed/Wiki10000/"
    data_file_prefix = "wiki10000"
    st_idx = str(800000)
    ed_idx = str(850000)
    os.system('CUDA_VISIBLE_DEVICES=0 PYTHONIOENCODING=utf-8 python3 QG_gpt2_generate.py  \
        --model_type gpt2 \
        --model_name_or_path ' + data_dir + 'output/QG/gpt2_question_generation/4epochs/2batchsize/ \
        --filename "' + processed_path + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.pkl" \
        --filecache "' + OUT_DIR + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.cache.qg.gpt2.pth" \
        --data_type augmented_sents \
        --output_file "' + OUT_DIR + data_file_prefix + '.qa.' + st_idx + '_' + ed_idx + '.qg.generated.gpt2.json"')

    processed_path = data_dir + "processed/Wiki10000/"
    data_file_prefix = "wiki10000"
    st_idx = str(850000)
    ed_idx = str(900000)
    os.system('CUDA_VISIBLE_DEVICES=0 PYTHONIOENCODING=utf-8 python3 QG_gpt2_generate.py  \
        --model_type gpt2 \
        --model_name_or_path ' + data_dir + 'output/QG/gpt2_question_generation/4epochs/2batchsize/ \
        --filename "' + processed_path + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.pkl" \
        --filecache "' + OUT_DIR + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.cache.qg.gpt2.pth" \
        --data_type augmented_sents \
        --output_file "' + OUT_DIR + data_file_prefix + '.qa.' + st_idx + '_' + ed_idx + '.qg.generated.gpt2.json"')

    processed_path = data_dir + "processed/Wiki10000/"
    data_file_prefix = "wiki10000"
    st_idx = str(900000)
    ed_idx = str(9500000)
    os.system('CUDA_VISIBLE_DEVICES=0 PYTHONIOENCODING=utf-8 python3 QG_gpt2_generate.py  \
        --model_type gpt2 \
        --model_name_or_path ' + data_dir + 'output/QG/gpt2_question_generation/4epochs/2batchsize/ \
        --filename "' + processed_path + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.pkl" \
        --filecache "' + OUT_DIR + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.cache.qg.gpt2.pth" \
        --data_type augmented_sents \
        --output_file "' + OUT_DIR + data_file_prefix + '.qa.' + st_idx + '_' + ed_idx + '.qg.generated.gpt2.json"')

    processed_path = data_dir + "processed/Wiki10000/"
    data_file_prefix = "wiki10000"
    st_idx = str(950000)
    ed_idx = str(1000000)
    os.system('CUDA_VISIBLE_DEVICES=0 PYTHONIOENCODING=utf-8 python3 QG_gpt2_generate.py  \
        --model_type gpt2 \
        --model_name_or_path ' + data_dir + 'output/QG/gpt2_question_generation/4epochs/2batchsize/ \
        --filename "' + processed_path + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.pkl" \
        --filecache "' + OUT_DIR + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.cache.qg.gpt2.pth" \
        --data_type augmented_sents \
        --output_file "' + OUT_DIR + data_file_prefix + '.qa.' + st_idx + '_' + ed_idx + '.qg.generated.gpt2.json"')
