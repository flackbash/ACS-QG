import os
import argparse
from polyaxon_client.tracking import get_outputs_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", default="", type=str,
                        help="Directory containing the data for the ACS-QG project. If given, skip preprocessing.")
    args = parser.parse_args()
    data_dir = args.data_dir

    OUT_DIR = get_outputs_path().rstrip("/") + "/"
    if data_dir:
        # skip preprocessing of data and look in specified directory for data
        directory = data_dir.rstrip("/") + "/"
        pre_process = ""
    else:
        # Perform preprocessing and save data in polyaxon output directory
        directory = OUT_DIR
        pre_process = "--not_processed_data"

    os.makedirs(os.path.join(OUT_DIR, "processed/SQuAD1.1-Zhou/"))
    os.system('python3 QG_main.py \
            --mode train \
            --batch_size 8 \
            --epochs 10 \
            --copy_type hard-oov \
            --copy_loss_type 1 \
            --use_style_info \
            --use_clue_info \
            -beam_size 20 \
            --use_refine_copy_tgt_src '
              + pre_process + ' \
            --checkpoint_dir ' + OUT_DIR + 'checkpoint/ \
            --train_examples_file ' + directory + 'processed/SQuAD1.1-Zhou/train-examples.pkl \
            --dev_examples_file ' + directory + 'processed/SQuAD1.1-Zhou/dev-examples.pkl \
            --test_examples_file ' + directory + 'processed/SQuAD1.1-Zhou/test-examples.pkl \
            --train_meta_file ' + directory + 'processed/SQuAD1.1-Zhou/train-meta.pkl \
            --dev_meta_file ' + directory + 'processed/SQuAD1.1-Zhou/dev-meta.pkl \
            --test_meta_file ' + directory + 'processed/SQuAD1.1-Zhou/test-meta.pkl \
            --train_eval_file ' + directory + 'processed/SQuAD1.1-Zhou/train-eval.pkl \
            --dev_eval_file ' + directory + 'processed/SQuAD1.1-Zhou/dev-eval.pkl \
            --test_eval_file ' + directory + 'processed/SQuAD1.1-Zhou/test-eval.pkl \
            --counters_file ' + directory + 'processed/SQuAD1.1-Zhou/counters.pkl \
            --emb_mats_file ' + directory + 'processed/SQuAD1.1-Zhou/emb_mats.pkl \
            --emb_dicts_file ' + directory + 'processed/SQuAD1.1-Zhou/emb_dicts.pkl \
            --related_words_dict_file ' + directory + 'processed/SQuAD1.1-Zhou/related_words_dict.pkl \
            --related_words_ids_mat_file ' + directory + 'processed/SQuAD1.1-Zhou/related_words_ids_mat.pkl ')
