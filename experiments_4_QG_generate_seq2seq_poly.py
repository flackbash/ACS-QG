# STEP 4: use trained FQG model to generate new QG data using augmented sentences
# run each code piece in one machine. process data in parallel.

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
    checkpoint_dir = data_dir + "output/checkpoint/"
    OUT_DIR = get_outputs_path().rstrip("/") + "/"

    # squad data
    processed_path = data_dir + "processed/SQuAD2.0/"
    data_file_prefix = "train"
    st_idx = str(0)
    ed_idx = str(50000)
    logger.info("run QG_augment_main.py over directory %s for indices %s-%s" % (processed_path, st_idx, ed_idx))
    os.system('CUDA_VISIBLE_DEVICES=0 python3 QG_augment_main.py \
            --not_processed_data  \
            --batch_size 8 \
            --epochs 10 \
            --copy_type hard-oov \
            --copy_loss_type 1 \
            --use_style_info \
            --use_clue_info \
            -beam_size 20 \
            --use_refine_copy_tgt_src \
            --da_augmented_sentences_file "' + processed_path + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.pkl" \
            --qg_augmented_sentences_file "' + OUT_DIR + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.processed.pkl" \
            --qg_result_file "' + OUT_DIR + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.output.txt" \
            --da_paragraphs_file "' + processed_path + data_file_prefix + '.paragraphs.txt" \
            --qa_data_file "' + OUT_DIR + data_file_prefix + '.qa.' + st_idx + '_' + ed_idx + '.txt" \
            --checkpoint_dir ' + checkpoint_dir)

    processed_path = data_dir + "processed/SQuAD2.0/"
    data_file_prefix = "train"
    st_idx = str(50000)
    ed_idx = str(92210)
    logger.info("run QG_augment_main.py over directory %s for indices %s-%s" % (processed_path, st_idx, ed_idx))
    os.system('CUDA_VISIBLE_DEVICES=0 python3 QG_augment_main.py \
            --not_processed_data  \
            --batch_size 8 \
            --epochs 10 \
            --copy_type hard-oov \
            --copy_loss_type 1 \
            --use_style_info \
            --use_clue_info \
            -beam_size 20 \
            --use_refine_copy_tgt_src \
            --da_augmented_sentences_file "' + processed_path + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.pkl" \
            --qg_augmented_sentences_file "' + OUT_DIR + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.processed.pkl" \
            --qg_result_file "' + OUT_DIR + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.output.txt" \
            --da_paragraphs_file "' + processed_path + data_file_prefix + '.paragraphs.txt" \
            --qa_data_file "' + OUT_DIR + data_file_prefix + '.qa.' + st_idx + '_' + ed_idx + '.txt" \
            --checkpoint_dir ' + checkpoint_dir)

    # wiki data
    processed_path = data_dir + "processed/Wiki10000/"
    data_file_prefix = "wiki10000"
    st_idx = str(0)
    ed_idx = str(50000)
    logger.info("run QG_augment_main.py over directory %s for indices %s-%s" % (processed_path, st_idx, ed_idx))
    os.system('CUDA_VISIBLE_DEVICES=0 python3 QG_augment_main.py \
            --not_processed_data  \
            --batch_size 8 \
            --epochs 10 \
            --copy_type hard-oov \
            --copy_loss_type 1 \
            --use_style_info \
            --use_clue_info \
            -beam_size 20 \
            --use_refine_copy_tgt_src \
            --da_augmented_sentences_file "' + processed_path + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.pkl" \
            --qg_augmented_sentences_file "' + OUT_DIR + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.processed.pkl" \
            --qg_result_file "' + OUT_DIR + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.output.txt" \
            --da_paragraphs_file "' + processed_path + data_file_prefix + '.paragraphs.txt" \
            --qa_data_file "' + OUT_DIR + data_file_prefix + '.qa.' + st_idx + '_' + ed_idx + '.txt" \
            --checkpoint_dir ' + checkpoint_dir)

    processed_path = data_dir + "processed/Wiki10000/"
    data_file_prefix = "wiki10000"
    st_idx = str(50000)
    ed_idx = str(100000)
    logger.info("run QG_augment_main.py over directory %s for indices %s-%s" % (processed_path, st_idx, ed_idx))
    os.system('CUDA_VISIBLE_DEVICES=0 python3 QG_augment_main.py \
            --not_processed_data  \
            --batch_size 8 \
            --epochs 10 \
            --copy_type hard-oov \
            --copy_loss_type 1 \
            --use_style_info \
            --use_clue_info \
            -beam_size 20 \
            --use_refine_copy_tgt_src \
            --da_augmented_sentences_file "' + processed_path + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.pkl" \
            --qg_augmented_sentences_file "' + OUT_DIR + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.processed.pkl" \
            --qg_result_file "' + OUT_DIR + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.output.txt" \
            --da_paragraphs_file "' + processed_path + data_file_prefix + '.paragraphs.txt" \
            --qa_data_file "' + OUT_DIR + data_file_prefix + '.qa.' + st_idx + '_' + ed_idx + '.txt" \
            --checkpoint_dir ' + checkpoint_dir)

    processed_path = data_dir + "processed/Wiki10000/"
    data_file_prefix = "wiki10000"
    st_idx = str(100000)
    ed_idx = str(150000)
    logger.info("run QG_augment_main.py over directory %s for indices %s-%s" % (processed_path, st_idx, ed_idx))
    os.system('CUDA_VISIBLE_DEVICES=0 python3 QG_augment_main.py \
            --not_processed_data  \
            --batch_size 8 \
            --epochs 10 \
            --copy_type hard-oov \
            --copy_loss_type 1 \
            --use_style_info \
            --use_clue_info \
            -beam_size 20 \
            --use_refine_copy_tgt_src \
            --da_augmented_sentences_file "' + processed_path + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.pkl" \
            --qg_augmented_sentences_file "' + OUT_DIR + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.processed.pkl" \
            --qg_result_file "' + OUT_DIR + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.output.txt" \
            --da_paragraphs_file "' + processed_path + data_file_prefix + '.paragraphs.txt" \
            --qa_data_file "' + OUT_DIR + data_file_prefix + '.qa.' + st_idx + '_' + ed_idx + '.txt" \
            --checkpoint_dir ' + checkpoint_dir)

    processed_path = data_dir + "processed/Wiki10000/"
    data_file_prefix = "wiki10000"
    st_idx = str(150000)
    ed_idx = str(200000)
    logger.info("run QG_augment_main.py over directory %s for indices %s-%s" % (processed_path, st_idx, ed_idx))
    os.system('CUDA_VISIBLE_DEVICES=0 python3 QG_augment_main.py \
            --not_processed_data  \
            --batch_size 8 \
            --epochs 10 \
            --copy_type hard-oov \
            --copy_loss_type 1 \
            --use_style_info \
            --use_clue_info \
            -beam_size 20 \
            --use_refine_copy_tgt_src \
            --da_augmented_sentences_file "' + processed_path + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.pkl" \
            --qg_augmented_sentences_file "' + OUT_DIR + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.processed.pkl" \
            --qg_result_file "' + OUT_DIR + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.output.txt" \
            --da_paragraphs_file "' + processed_path + data_file_prefix + '.paragraphs.txt" \
            --qa_data_file "' + OUT_DIR + data_file_prefix + '.qa.' + st_idx + '_' + ed_idx + '.txt" \
            --checkpoint_dir ' + checkpoint_dir)

    processed_path = data_dir + "processed/Wiki10000/"
    data_file_prefix = "wiki10000"
    st_idx = str(200000)
    ed_idx = str(250000)
    logger.info("run QG_augment_main.py over directory %s for indices %s-%s" % (processed_path, st_idx, ed_idx))
    os.system('CUDA_VISIBLE_DEVICES=0 python3 QG_augment_main.py \
            --not_processed_data  \
            --batch_size 8 \
            --epochs 10 \
            --copy_type hard-oov \
            --copy_loss_type 1 \
            --use_style_info \
            --use_clue_info \
            -beam_size 20 \
            --use_refine_copy_tgt_src \
            --da_augmented_sentences_file "' + processed_path + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.pkl" \
            --qg_augmented_sentences_file "' + OUT_DIR + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.processed.pkl" \
            --qg_result_file "' + OUT_DIR + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.output.txt" \
            --da_paragraphs_file "' + processed_path + data_file_prefix + '.paragraphs.txt" \
            --qa_data_file "' + OUT_DIR + data_file_prefix + '.qa.' + st_idx + '_' + ed_idx + '.txt" \
            --checkpoint_dir ' + checkpoint_dir)

    processed_path = data_dir + "processed/Wiki10000/"
    data_file_prefix = "wiki10000"
    st_idx = str(250000)
    ed_idx = str(300000)
    logger.info("run QG_augment_main.py over directory %s for indices %s-%s" % (processed_path, st_idx, ed_idx))
    os.system('CUDA_VISIBLE_DEVICES=0 python3 QG_augment_main.py \
            --not_processed_data  \
            --batch_size 8 \
            --epochs 10 \
            --copy_type hard-oov \
            --copy_loss_type 1 \
            --use_style_info \
            --use_clue_info \
            -beam_size 20 \
            --use_refine_copy_tgt_src \
            --da_augmented_sentences_file "' + processed_path + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.pkl" \
            --qg_augmented_sentences_file "' + OUT_DIR + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.processed.pkl" \
            --qg_result_file "' + OUT_DIR + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.output.txt" \
            --da_paragraphs_file "' + processed_path + data_file_prefix + '.paragraphs.txt" \
            --qa_data_file "' + OUT_DIR + data_file_prefix + '.qa.' + st_idx + '_' + ed_idx + '.txt" \
            --checkpoint_dir ' + checkpoint_dir)

    processed_path = data_dir + "processed/Wiki10000/"
    data_file_prefix = "wiki10000"
    st_idx = str(300000)
    ed_idx = str(350000)
    logger.info("run QG_augment_main.py over directory %s for indices %s-%s" % (processed_path, st_idx, ed_idx))
    os.system('CUDA_VISIBLE_DEVICES=0 python3 QG_augment_main.py \
            --not_processed_data  \
            --batch_size 8 \
            --epochs 10 \
            --copy_type hard-oov \
            --copy_loss_type 1 \
            --use_style_info \
            --use_clue_info \
            -beam_size 20 \
            --use_refine_copy_tgt_src \
            --da_augmented_sentences_file "' + processed_path + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.pkl" \
            --qg_augmented_sentences_file "' + OUT_DIR + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.processed.pkl" \
            --qg_result_file "' + OUT_DIR + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.output.txt" \
            --da_paragraphs_file "' + processed_path + data_file_prefix + '.paragraphs.txt" \
            --qa_data_file "' + OUT_DIR + data_file_prefix + '.qa.' + st_idx + '_' + ed_idx + '.txt" \
            --checkpoint_dir ' + checkpoint_dir)

    processed_path = data_dir + "processed/Wiki10000/"
    data_file_prefix = "wiki10000"
    st_idx = str(350000)
    ed_idx = str(400000)
    logger.info("run QG_augment_main.py over directory %s for indices %s-%s" % (processed_path, st_idx, ed_idx))
    os.system('CUDA_VISIBLE_DEVICES=0 python3 QG_augment_main.py \
            --not_processed_data  \
            --batch_size 8 \
            --epochs 10 \
            --copy_type hard-oov \
            --copy_loss_type 1 \
            --use_style_info \
            --use_clue_info \
            -beam_size 20 \
            --use_refine_copy_tgt_src \
            --da_augmented_sentences_file "' + processed_path + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.pkl" \
            --qg_augmented_sentences_file "' + OUT_DIR + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.processed.pkl" \
            --qg_result_file "' + OUT_DIR + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.output.txt" \
            --da_paragraphs_file "' + processed_path + data_file_prefix + '.paragraphs.txt" \
            --qa_data_file "' + OUT_DIR + data_file_prefix + '.qa.' + st_idx + '_' + ed_idx + '.txt" \
            --checkpoint_dir ' + checkpoint_dir)

    processed_path = data_dir + "processed/Wiki10000/"
    data_file_prefix = "wiki10000"
    st_idx = str(400000)
    ed_idx = str(450000)
    logger.info("run QG_augment_main.py over directory %s for indices %s-%s" % (processed_path, st_idx, ed_idx))
    os.system('CUDA_VISIBLE_DEVICES=0 python3 QG_augment_main.py \
            --not_processed_data  \
            --batch_size 8 \
            --epochs 10 \
            --copy_type hard-oov \
            --copy_loss_type 1 \
            --use_style_info \
            --use_clue_info \
            -beam_size 20 \
            --use_refine_copy_tgt_src \
            --da_augmented_sentences_file "' + processed_path + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.pkl" \
            --qg_augmented_sentences_file "' + OUT_DIR + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.processed.pkl" \
            --qg_result_file "' + OUT_DIR + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.output.txt" \
            --da_paragraphs_file "' + processed_path + data_file_prefix + '.paragraphs.txt" \
            --qa_data_file "' + OUT_DIR + data_file_prefix + '.qa.' + st_idx + '_' + ed_idx + '.txt" \
            --checkpoint_dir ' + checkpoint_dir)

    processed_path = data_dir + "processed/Wiki10000/"
    data_file_prefix = "wiki10000"
    st_idx = str(450000)
    ed_idx = str(500000)
    logger.info("run QG_augment_main.py over directory %s for indices %s-%s" % (processed_path, st_idx, ed_idx))
    os.system('CUDA_VISIBLE_DEVICES=0 python3 QG_augment_main.py \
            --not_processed_data  \
            --batch_size 8 \
            --epochs 10 \
            --copy_type hard-oov \
            --copy_loss_type 1 \
            --use_style_info \
            --use_clue_info \
            -beam_size 20 \
            --use_refine_copy_tgt_src \
            --da_augmented_sentences_file "' + processed_path + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.pkl" \
            --qg_augmented_sentences_file "' + OUT_DIR + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.processed.pkl" \
            --qg_result_file "' + OUT_DIR + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.output.txt" \
            --da_paragraphs_file "' + processed_path + data_file_prefix + '.paragraphs.txt" \
            --qa_data_file "' + OUT_DIR + data_file_prefix + '.qa.' + st_idx + '_' + ed_idx + '.txt" \
            --checkpoint_dir ' + checkpoint_dir)

    processed_path = data_dir + "processed/Wiki10000/"
    data_file_prefix = "wiki10000"
    st_idx = str(500000)
    ed_idx = str(550000)
    logger.info("run QG_augment_main.py over directory %s for indices %s-%s" % (processed_path, st_idx, ed_idx))
    os.system('CUDA_VISIBLE_DEVICES=0 python3 QG_augment_main.py \
            --not_processed_data  \
            --batch_size 8 \
            --epochs 10 \
            --copy_type hard-oov \
            --copy_loss_type 1 \
            --use_style_info \
            --use_clue_info \
            -beam_size 20 \
            --use_refine_copy_tgt_src \
            --da_augmented_sentences_file "' + processed_path + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.pkl" \
            --qg_augmented_sentences_file "' + OUT_DIR + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.processed.pkl" \
            --qg_result_file "' + OUT_DIR + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.output.txt" \
            --da_paragraphs_file "' + processed_path + data_file_prefix + '.paragraphs.txt" \
            --qa_data_file "' + OUT_DIR + data_file_prefix + '.qa.' + st_idx + '_' + ed_idx + '.txt" \
            --checkpoint_dir ' + checkpoint_dir)

    processed_path = data_dir + "processed/Wiki10000/"
    data_file_prefix = "wiki10000"
    st_idx = str(550000)
    ed_idx = str(600000)
    logger.info("run QG_augment_main.py over directory %s for indices %s-%s" % (processed_path, st_idx, ed_idx))
    os.system('CUDA_VISIBLE_DEVICES=0 python3 QG_augment_main.py \
            --not_processed_data  \
            --batch_size 8 \
            --epochs 10 \
            --copy_type hard-oov \
            --copy_loss_type 1 \
            --use_style_info \
            --use_clue_info \
            -beam_size 20 \
            --use_refine_copy_tgt_src \
            --da_augmented_sentences_file "' + processed_path + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.pkl" \
            --qg_augmented_sentences_file "' + OUT_DIR + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.processed.pkl" \
            --qg_result_file "' + OUT_DIR + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.output.txt" \
            --da_paragraphs_file "' + processed_path + data_file_prefix + '.paragraphs.txt" \
            --qa_data_file "' + OUT_DIR + data_file_prefix + '.qa.' + st_idx + '_' + ed_idx + '.txt" \
            --checkpoint_dir ' + checkpoint_dir)

    processed_path = data_dir + "processed/Wiki10000/"
    data_file_prefix = "wiki10000"
    st_idx = str(600000)
    ed_idx = str(650000)
    logger.info("run QG_augment_main.py over directory %s for indices %s-%s" % (processed_path, st_idx, ed_idx))
    os.system('CUDA_VISIBLE_DEVICES=0 python3 QG_augment_main.py \
            --not_processed_data  \
            --batch_size 8 \
            --epochs 10 \
            --copy_type hard-oov \
            --copy_loss_type 1 \
            --use_style_info \
            --use_clue_info \
            -beam_size 20 \
            --use_refine_copy_tgt_src \
            --da_augmented_sentences_file "' + processed_path + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.pkl" \
            --qg_augmented_sentences_file "' + OUT_DIR + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.processed.pkl" \
            --qg_result_file "' + OUT_DIR + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.output.txt" \
            --da_paragraphs_file "' + processed_path + data_file_prefix + '.paragraphs.txt" \
            --qa_data_file "' + OUT_DIR + data_file_prefix + '.qa.' + st_idx + '_' + ed_idx + '.txt" \
            --checkpoint_dir ' + checkpoint_dir)

    processed_path = data_dir + "processed/Wiki10000/"
    data_file_prefix = "wiki10000"
    st_idx = str(650000)
    ed_idx = str(700000)
    logger.info("run QG_augment_main.py over directory %s for indices %s-%s" % (processed_path, st_idx, ed_idx))
    os.system('CUDA_VISIBLE_DEVICES=0 python3 QG_augment_main.py \
            --not_processed_data  \
            --batch_size 8 \
            --epochs 10 \
            --copy_type hard-oov \
            --copy_loss_type 1 \
            --use_style_info \
            --use_clue_info \
            -beam_size 20 \
            --use_refine_copy_tgt_src \
            --da_augmented_sentences_file "' + processed_path + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.pkl" \
            --qg_augmented_sentences_file "' + OUT_DIR + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.processed.pkl" \
            --qg_result_file "' + OUT_DIR + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.output.txt" \
            --da_paragraphs_file "' + processed_path + data_file_prefix + '.paragraphs.txt" \
            --qa_data_file "' + OUT_DIR + data_file_prefix + '.qa.' + st_idx + '_' + ed_idx + '.txt" \
            --checkpoint_dir ' + checkpoint_dir)

    processed_path = data_dir + "processed/Wiki10000/"
    data_file_prefix = "wiki10000"
    st_idx = str(700000)
    ed_idx = str(750000)
    logger.info("run QG_augment_main.py over directory %s for indices %s-%s" % (processed_path, st_idx, ed_idx))
    os.system('CUDA_VISIBLE_DEVICES=0 python3 QG_augment_main.py \
            --not_processed_data  \
            --batch_size 8 \
            --epochs 10 \
            --copy_type hard-oov \
            --copy_loss_type 1 \
            --use_style_info \
            --use_clue_info \
            -beam_size 20 \
            --use_refine_copy_tgt_src \
            --da_augmented_sentences_file "' + processed_path + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.pkl" \
            --qg_augmented_sentences_file "' + OUT_DIR + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.processed.pkl" \
            --qg_result_file "' + OUT_DIR + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.output.txt" \
            --da_paragraphs_file "' + processed_path + data_file_prefix + '.paragraphs.txt" \
            --qa_data_file "' + OUT_DIR + data_file_prefix + '.qa.' + st_idx + '_' + ed_idx + '.txt" \
            --checkpoint_dir ' + checkpoint_dir)

    processed_path = data_dir + "processed/Wiki10000/"
    data_file_prefix = "wiki10000"
    st_idx = str(750000)
    ed_idx = str(800000)
    logger.info("run QG_augment_main.py over directory %s for indices %s-%s" % (processed_path, st_idx, ed_idx))
    os.system('CUDA_VISIBLE_DEVICES=0 python3 QG_augment_main.py \
            --not_processed_data  \
            --batch_size 8 \
            --epochs 10 \
            --copy_type hard-oov \
            --copy_loss_type 1 \
            --use_style_info \
            --use_clue_info \
            -beam_size 20 \
            --use_refine_copy_tgt_src \
            --da_augmented_sentences_file "' + processed_path + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.pkl" \
            --qg_augmented_sentences_file "' + OUT_DIR + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.processed.pkl" \
            --qg_result_file "' + OUT_DIR + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.output.txt" \
            --da_paragraphs_file "' + processed_path + data_file_prefix + '.paragraphs.txt" \
            --qa_data_file "' + OUT_DIR + data_file_prefix + '.qa.' + st_idx + '_' + ed_idx + '.txt" \
            --checkpoint_dir ' + checkpoint_dir)

    processed_path = data_dir + "processed/Wiki10000/"
    data_file_prefix = "wiki10000"
    st_idx = str(800000)
    ed_idx = str(850000)
    logger.info("run QG_augment_main.py over directory %s for indices %s-%s" % (processed_path, st_idx, ed_idx))
    os.system('CUDA_VISIBLE_DEVICES=0 python3 QG_augment_main.py \
            --not_processed_data  \
            --batch_size 8 \
            --epochs 10 \
            --copy_type hard-oov \
            --copy_loss_type 1 \
            --use_style_info \
            --use_clue_info \
            -beam_size 20 \
            --use_refine_copy_tgt_src \
            --da_augmented_sentences_file "' + processed_path + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.pkl" \
            --qg_augmented_sentences_file "' + OUT_DIR + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.processed.pkl" \
            --qg_result_file "' + OUT_DIR + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.output.txt" \
            --da_paragraphs_file "' + processed_path + data_file_prefix + '.paragraphs.txt" \
            --qa_data_file "' + OUT_DIR + data_file_prefix + '.qa.' + st_idx + '_' + ed_idx + '.txt" \
            --checkpoint_dir ' + checkpoint_dir)

    processed_path = data_dir + "processed/Wiki10000/"
    data_file_prefix = "wiki10000"
    st_idx = str(850000)
    ed_idx = str(900000)
    logger.info("run QG_augment_main.py over directory %s for indices %s-%s" % (processed_path, st_idx, ed_idx))
    os.system('CUDA_VISIBLE_DEVICES=0 python3 QG_augment_main.py \
            --not_processed_data  \
            --batch_size 8 \
            --epochs 10 \
            --copy_type hard-oov \
            --copy_loss_type 1 \
            --use_style_info \
            --use_clue_info \
            -beam_size 20 \
            --use_refine_copy_tgt_src \
            --da_augmented_sentences_file "' + processed_path + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.pkl" \
            --qg_augmented_sentences_file "' + OUT_DIR + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.processed.pkl" \
            --qg_result_file "' + OUT_DIR + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.output.txt" \
            --da_paragraphs_file "' + processed_path + data_file_prefix + '.paragraphs.txt" \
            --qa_data_file "' + OUT_DIR + data_file_prefix + '.qa.' + st_idx + '_' + ed_idx + '.txt" \
            --checkpoint_dir ' + checkpoint_dir)

    processed_path = data_dir + "processed/Wiki10000/"
    data_file_prefix = "wiki10000"
    st_idx = str(900000)
    ed_idx = str(950000)
    logger.info("run QG_augment_main.py over directory %s for indices %s-%s" % (processed_path, st_idx, ed_idx))
    os.system('CUDA_VISIBLE_DEVICES=0 python3 QG_augment_main.py \
            --not_processed_data  \
            --batch_size 8 \
            --epochs 10 \
            --copy_type hard-oov \
            --copy_loss_type 1 \
            --use_style_info \
            --use_clue_info \
            -beam_size 20 \
            --use_refine_copy_tgt_src \
            --da_augmented_sentences_file "' + processed_path + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.pkl" \
            --qg_augmented_sentences_file "' + OUT_DIR + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.processed.pkl" \
            --qg_result_file "' + OUT_DIR + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.output.txt" \
            --da_paragraphs_file "' + processed_path + data_file_prefix + '.paragraphs.txt" \
            --qa_data_file "' + OUT_DIR + data_file_prefix + '.qa.' + st_idx + '_' + ed_idx + '.txt" \
            --checkpoint_dir ' + checkpoint_dir)

    processed_path = data_dir + "processed/Wiki10000/"
    data_file_prefix = "wiki10000"
    st_idx = str(950000)
    ed_idx = str(1000000)
    logger.info("run QG_augment_main.py over directory %s for indices %s-%s" % (processed_path, st_idx, ed_idx))
    os.system('CUDA_VISIBLE_DEVICES=0 python3 QG_augment_main.py \
            --not_processed_data  \
            --batch_size 8 \
            --epochs 10 \
            --copy_type hard-oov \
            --copy_loss_type 1 \
            --use_style_info \
            --use_clue_info \
            -beam_size 20 \
            --use_refine_copy_tgt_src \
            --da_augmented_sentences_file "' + processed_path + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.pkl" \
            --qg_augmented_sentences_file "' + OUT_DIR + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.processed.pkl" \
            --qg_result_file "' + OUT_DIR + data_file_prefix + '.sentences.augmented.' + st_idx + '_' + ed_idx + '.output.txt" \
            --da_paragraphs_file "' + processed_path + data_file_prefix + '.paragraphs.txt" \
            --qa_data_file "' + OUT_DIR + data_file_prefix + '.qa.' + st_idx + '_' + ed_idx + '.txt" \
            --checkpoint_dir ' + checkpoint_dir)
