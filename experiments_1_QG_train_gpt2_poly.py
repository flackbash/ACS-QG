# Note: After the completion of this polyaxon experiment, move the corresponding output files or directories to
# <data_directory>/output/QG/gpt2_question_generation/
import os
import argparse
from polyaxon_client.tracking import get_outputs_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, help="Directory containing the data for the ACS-QG project")
    args = parser.parse_args()

    data_dir = args.data_dir.rstrip("/") + "/"
    OUT_DIR = get_outputs_path().rstrip("/")

    os.system('CUDA_VISIBLE_DEVICES=1 python3 QG_gpt2_train.py \
        --eval_before_start \
        --n_epochs 4 \
        --model_name_or_path gpt2 \
        --output_dir ' + OUT_DIR + ' \
        --train_dataset_path ' + data_dir + 'original/SQuAD1.1-Zhou/train.txt \
        --dev_dataset_path ' + data_dir + 'original/SQuAD1.1-Zhou/dev.txt')
