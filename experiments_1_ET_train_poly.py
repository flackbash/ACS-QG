import os
import argparse
from polyaxon_client.tracking import get_outputs_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, help="Directory containing the data for the ACS-QG project")
    args = parser.parse_args()

    data_dir = args.data_dir.rstrip("/") + "/"
    OUT_DIR = get_outputs_path().rstrip("/") + "/"

    os.system('python3 run_glue.py \
            --model_type xlnet \
            --model_name_or_path xlnet-base-cased \
            --task_name MRPC \
            --do_train \
            --do_eval \
            --do_lower_case \
            --data_dir ' + data_dir + 'glue_data/MRPC/ \
            --max_seq_length 128 \
            --per_gpu_eval_batch_size=8   \
            --per_gpu_train_batch_size=8   \
            --learning_rate 2e-5 \
            --num_train_epochs 1.0 \
            --output_dir ' + OUT_DIR + ' \
            --overwrite_output_dir')
