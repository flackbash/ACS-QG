# STEP 5: use trained entailment model to append entailment score column

output_path="/home/Datasets/processed/SQuAD2.0/"
data_file_prefix="train"
st_idx=0
ed_idx=50000
python3 run_glue.py \
        --model_type xlnet \
        --model_name_or_path xlnet-base-cased \
        --task_name MRPC \
        --do_test \
        --do_lower_case \
        --data_dir /home/Datasets/glue_data/MRPC/ \
        --max_seq_length 128 \
        --per_gpu_eval_batch_size=8   \
        --per_gpu_train_batch_size=8   \
        --learning_rate 2e-5 \
        --num_train_epochs 1.0 \
        --output_dir /home/Datasets/output/ET/xlnet-base-cased/ \
        --overwrite_output_dir \
        --context_question_answer_file "${output_path}${data_file_prefix}.qa.${st_idx}_${ed_idx}.uniq.postprocessed.txt" \
        --context_question_answer_columns 3 2 4 \
        --context_question_answer_score_file "${output_path}${data_file_prefix}.qa.${st_idx}_${ed_idx}.entail.txt"


output_path="/home/Datasets/processed/SQuAD2.0/"
data_file_prefix="train"
st_idx=50000
ed_idx=92210
python3 run_glue.py \
        --model_type xlnet \
        --model_name_or_path xlnet-base-cased \
        --task_name MRPC \
        --do_test \
        --do_lower_case \
        --data_dir /home/Datasets/glue_data/MRPC/ \
        --max_seq_length 128 \
        --per_gpu_eval_batch_size=8   \
        --per_gpu_train_batch_size=8   \
        --learning_rate 2e-5 \
        --num_train_epochs 1.0 \
        --output_dir /home/Datasets/output/ET/xlnet-base-cased/ \
        --overwrite_output_dir \
        --context_question_answer_file "${output_path}${data_file_prefix}.qa.${st_idx}_${ed_idx}.uniq.postprocessed.txt" \
        --context_question_answer_columns 3 2 4 \
        --context_question_answer_score_file "${output_path}${data_file_prefix}.qa.${st_idx}_${ed_idx}.entail.txt"


output_path="/home/Datasets/processed/Wiki10000/"
data_file_prefix="wiki10000"
st_idx=0
ed_idx=50000
python3 run_glue.py \
        --model_type xlnet \
        --model_name_or_path xlnet-base-cased \
        --task_name MRPC \
        --do_test \
        --do_lower_case \
        --data_dir /home/Datasets/glue_data/MRPC/ \
        --max_seq_length 128 \
        --per_gpu_eval_batch_size=8   \
        --per_gpu_train_batch_size=8   \
        --learning_rate 2e-5 \
        --num_train_epochs 1.0 \
        --output_dir /home/Datasets/output/ET/xlnet-base-cased/ \
        --overwrite_output_dir \
        --context_question_answer_file "${output_path}${data_file_prefix}.qa.${st_idx}_${ed_idx}.uniq.postprocessed.txt" \
        --context_question_answer_columns 3 2 4 \
        --context_question_answer_score_file "${output_path}${data_file_prefix}.qa.${st_idx}_${ed_idx}.entail.txt"


output_path="/home/Datasets/processed/Wiki10000/"
data_file_prefix="wiki10000"
st_idx=50000
ed_idx=100000
python3 run_glue.py \
        --model_type xlnet \
        --model_name_or_path xlnet-base-cased \
        --task_name MRPC \
        --do_test \
        --do_lower_case \
        --data_dir /home/Datasets/glue_data/MRPC/ \
        --max_seq_length 128 \
        --per_gpu_eval_batch_size=8   \
        --per_gpu_train_batch_size=8   \
        --learning_rate 2e-5 \
        --num_train_epochs 1.0 \
        --output_dir /home/Datasets/output/ET/xlnet-base-cased/ \
        --overwrite_output_dir \
        --context_question_answer_file "${output_path}${data_file_prefix}.qa.${st_idx}_${ed_idx}.uniq.postprocessed.txt" \
        --context_question_answer_columns 3 2 4 \
        --context_question_answer_score_file "${output_path}${data_file_prefix}.qa.${st_idx}_${ed_idx}.entail.txt"


output_path="/home/Datasets/processed/Wiki10000/"
data_file_prefix="wiki10000"
st_idx=100000
ed_idx=150000
python3 run_glue.py \
        --model_type xlnet \
        --model_name_or_path xlnet-base-cased \
        --task_name MRPC \
        --do_test \
        --do_lower_case \
        --data_dir /home/Datasets/glue_data/MRPC/ \
        --max_seq_length 128 \
        --per_gpu_eval_batch_size=8   \
        --per_gpu_train_batch_size=8   \
        --learning_rate 2e-5 \
        --num_train_epochs 1.0 \
        --output_dir /home/Datasets/output/ET/xlnet-base-cased/ \
        --overwrite_output_dir \
        --context_question_answer_file "${output_path}${data_file_prefix}.qa.${st_idx}_${ed_idx}.uniq.postprocessed.txt" \
        --context_question_answer_columns 3 2 4 \
        --context_question_answer_score_file "${output_path}${data_file_prefix}.qa.${st_idx}_${ed_idx}.entail.txt"


output_path="/home/Datasets/processed/Wiki10000/"
data_file_prefix="wiki10000"
st_idx=150000
ed_idx=200000
python3 run_glue.py \
        --model_type xlnet \
        --model_name_or_path xlnet-base-cased \
        --task_name MRPC \
        --do_test \
        --do_lower_case \
        --data_dir /home/Datasets/glue_data/MRPC/ \
        --max_seq_length 128 \
        --per_gpu_eval_batch_size=8   \
        --per_gpu_train_batch_size=8   \
        --learning_rate 2e-5 \
        --num_train_epochs 1.0 \
        --output_dir /home/Datasets/output/ET/xlnet-base-cased/ \
        --overwrite_output_dir \
        --context_question_answer_file "${output_path}${data_file_prefix}.qa.${st_idx}_${ed_idx}.uniq.postprocessed.txt" \
        --context_question_answer_columns 3 2 4 \
        --context_question_answer_score_file "${output_path}${data_file_prefix}.qa.${st_idx}_${ed_idx}.entail.txt"


output_path="/home/Datasets/processed/Wiki10000/"
data_file_prefix="wiki10000"
st_idx=200000
ed_idx=250000
python3 run_glue.py \
        --model_type xlnet \
        --model_name_or_path xlnet-base-cased \
        --task_name MRPC \
        --do_test \
        --do_lower_case \
        --data_dir /home/Datasets/glue_data/MRPC/ \
        --max_seq_length 128 \
        --per_gpu_eval_batch_size=8   \
        --per_gpu_train_batch_size=8   \
        --learning_rate 2e-5 \
        --num_train_epochs 1.0 \
        --output_dir /home/Datasets/output/ET/xlnet-base-cased/ \
        --overwrite_output_dir \
        --context_question_answer_file "${output_path}${data_file_prefix}.qa.${st_idx}_${ed_idx}.uniq.postprocessed.txt" \
        --context_question_answer_columns 3 2 4 \
        --context_question_answer_score_file "${output_path}${data_file_prefix}.qa.${st_idx}_${ed_idx}.entail.txt"


output_path="/home/Datasets/processed/Wiki10000/"
data_file_prefix="wiki10000"
st_idx=250000
ed_idx=300000
python3 run_glue.py \
        --model_type xlnet \
        --model_name_or_path xlnet-base-cased \
        --task_name MRPC \
        --do_test \
        --do_lower_case \
        --data_dir /home/Datasets/glue_data/MRPC/ \
        --max_seq_length 128 \
        --per_gpu_eval_batch_size=8   \
        --per_gpu_train_batch_size=8   \
        --learning_rate 2e-5 \
        --num_train_epochs 1.0 \
        --output_dir /home/Datasets/output/ET/xlnet-base-cased/ \
        --overwrite_output_dir \
        --context_question_answer_file "${output_path}${data_file_prefix}.qa.${st_idx}_${ed_idx}.uniq.postprocessed.txt" \
        --context_question_answer_columns 3 2 4 \
        --context_question_answer_score_file "${output_path}${data_file_prefix}.qa.${st_idx}_${ed_idx}.entail.txt"


output_path="/home/Datasets/processed/Wiki10000/"
data_file_prefix="wiki10000"
st_idx=300000
ed_idx=350000
python3 run_glue.py \
        --model_type xlnet \
        --model_name_or_path xlnet-base-cased \
        --task_name MRPC \
        --do_test \
        --do_lower_case \
        --data_dir /home/Datasets/glue_data/MRPC/ \
        --max_seq_length 128 \
        --per_gpu_eval_batch_size=8   \
        --per_gpu_train_batch_size=8   \
        --learning_rate 2e-5 \
        --num_train_epochs 1.0 \
        --output_dir /home/Datasets/output/ET/xlnet-base-cased/ \
        --overwrite_output_dir \
        --context_question_answer_file "${output_path}${data_file_prefix}.qa.${st_idx}_${ed_idx}.uniq.postprocessed.txt" \
        --context_question_answer_columns 3 2 4 \
        --context_question_answer_score_file "${output_path}${data_file_prefix}.qa.${st_idx}_${ed_idx}.entail.txt"


output_path="/home/Datasets/processed/Wiki10000/"
data_file_prefix="wiki10000"
st_idx=350000
ed_idx=400000
python3 run_glue.py \
        --model_type xlnet \
        --model_name_or_path xlnet-base-cased \
        --task_name MRPC \
        --do_test \
        --do_lower_case \
        --data_dir /home/Datasets/glue_data/MRPC/ \
        --max_seq_length 128 \
        --per_gpu_eval_batch_size=8   \
        --per_gpu_train_batch_size=8   \
        --learning_rate 2e-5 \
        --num_train_epochs 1.0 \
        --output_dir /home/Datasets/output/ET/xlnet-base-cased/ \
        --overwrite_output_dir \
        --context_question_answer_file "${output_path}${data_file_prefix}.qa.${st_idx}_${ed_idx}.uniq.postprocessed.txt" \
        --context_question_answer_columns 3 2 4 \
        --context_question_answer_score_file "${output_path}${data_file_prefix}.qa.${st_idx}_${ed_idx}.entail.txt"


output_path="/home/Datasets/processed/Wiki10000/"
data_file_prefix="wiki10000"
st_idx=400000
ed_idx=450000
python3 run_glue.py \
        --model_type xlnet \
        --model_name_or_path xlnet-base-cased \
        --task_name MRPC \
        --do_test \
        --do_lower_case \
        --data_dir /home/Datasets/glue_data/MRPC/ \
        --max_seq_length 128 \
        --per_gpu_eval_batch_size=8   \
        --per_gpu_train_batch_size=8   \
        --learning_rate 2e-5 \
        --num_train_epochs 1.0 \
        --output_dir /home/Datasets/output/ET/xlnet-base-cased/ \
        --overwrite_output_dir \
        --context_question_answer_file "${output_path}${data_file_prefix}.qa.${st_idx}_${ed_idx}.uniq.postprocessed.txt" \
        --context_question_answer_columns 3 2 4 \
        --context_question_answer_score_file "${output_path}${data_file_prefix}.qa.${st_idx}_${ed_idx}.entail.txt"


output_path="/home/Datasets/processed/Wiki10000/"
data_file_prefix="wiki10000"
st_idx=450000
ed_idx=500000
python3 run_glue.py \
        --model_type xlnet \
        --model_name_or_path xlnet-base-cased \
        --task_name MRPC \
        --do_test \
        --do_lower_case \
        --data_dir /home/Datasets/glue_data/MRPC/ \
        --max_seq_length 128 \
        --per_gpu_eval_batch_size=8   \
        --per_gpu_train_batch_size=8   \
        --learning_rate 2e-5 \
        --num_train_epochs 1.0 \
        --output_dir /home/Datasets/output/ET/xlnet-base-cased/ \
        --overwrite_output_dir \
        --context_question_answer_file "${output_path}${data_file_prefix}.qa.${st_idx}_${ed_idx}.uniq.postprocessed.txt" \
        --context_question_answer_columns 3 2 4 \
        --context_question_answer_score_file "${output_path}${data_file_prefix}.qa.${st_idx}_${ed_idx}.entail.txt"


output_path="/home/Datasets/processed/Wiki10000/"
data_file_prefix="wiki10000"
st_idx=500000
ed_idx=550000
python3 run_glue.py \
        --model_type xlnet \
        --model_name_or_path xlnet-base-cased \
        --task_name MRPC \
        --do_test \
        --do_lower_case \
        --data_dir /home/Datasets/glue_data/MRPC/ \
        --max_seq_length 128 \
        --per_gpu_eval_batch_size=8   \
        --per_gpu_train_batch_size=8   \
        --learning_rate 2e-5 \
        --num_train_epochs 1.0 \
        --output_dir /home/Datasets/output/ET/xlnet-base-cased/ \
        --overwrite_output_dir \
        --context_question_answer_file "${output_path}${data_file_prefix}.qa.${st_idx}_${ed_idx}.uniq.postprocessed.txt" \
        --context_question_answer_columns 3 2 4 \
        --context_question_answer_score_file "${output_path}${data_file_prefix}.qa.${st_idx}_${ed_idx}.entail.txt"


output_path="/home/Datasets/processed/Wiki10000/"
data_file_prefix="wiki10000"
st_idx=550000
ed_idx=600000
python3 run_glue.py \
        --model_type xlnet \
        --model_name_or_path xlnet-base-cased \
        --task_name MRPC \
        --do_test \
        --do_lower_case \
        --data_dir /home/Datasets/glue_data/MRPC/ \
        --max_seq_length 128 \
        --per_gpu_eval_batch_size=8   \
        --per_gpu_train_batch_size=8   \
        --learning_rate 2e-5 \
        --num_train_epochs 1.0 \
        --output_dir /home/Datasets/output/ET/xlnet-base-cased/ \
        --overwrite_output_dir \
        --context_question_answer_file "${output_path}${data_file_prefix}.qa.${st_idx}_${ed_idx}.uniq.postprocessed.txt" \
        --context_question_answer_columns 3 2 4 \
        --context_question_answer_score_file "${output_path}${data_file_prefix}.qa.${st_idx}_${ed_idx}.entail.txt"


output_path="/home/Datasets/processed/Wiki10000/"
data_file_prefix="wiki10000"
st_idx=600000
ed_idx=650000
python3 run_glue.py \
        --model_type xlnet \
        --model_name_or_path xlnet-base-cased \
        --task_name MRPC \
        --do_test \
        --do_lower_case \
        --data_dir /home/Datasets/glue_data/MRPC/ \
        --max_seq_length 128 \
        --per_gpu_eval_batch_size=8   \
        --per_gpu_train_batch_size=8   \
        --learning_rate 2e-5 \
        --num_train_epochs 1.0 \
        --output_dir /home/Datasets/output/ET/xlnet-base-cased/ \
        --overwrite_output_dir \
        --context_question_answer_file "${output_path}${data_file_prefix}.qa.${st_idx}_${ed_idx}.uniq.postprocessed.txt" \
        --context_question_answer_columns 3 2 4 \
        --context_question_answer_score_file "${output_path}${data_file_prefix}.qa.${st_idx}_${ed_idx}.entail.txt"


output_path="/home/Datasets/processed/Wiki10000/"
data_file_prefix="wiki10000"
st_idx=650000
ed_idx=700000
python3 run_glue.py \
        --model_type xlnet \
        --model_name_or_path xlnet-base-cased \
        --task_name MRPC \
        --do_test \
        --do_lower_case \
        --data_dir /home/Datasets/glue_data/MRPC/ \
        --max_seq_length 128 \
        --per_gpu_eval_batch_size=8   \
        --per_gpu_train_batch_size=8   \
        --learning_rate 2e-5 \
        --num_train_epochs 1.0 \
        --output_dir /home/Datasets/output/ET/xlnet-base-cased/ \
        --overwrite_output_dir \
        --context_question_answer_file "${output_path}${data_file_prefix}.qa.${st_idx}_${ed_idx}.uniq.postprocessed.txt" \
        --context_question_answer_columns 3 2 4 \
        --context_question_answer_score_file "${output_path}${data_file_prefix}.qa.${st_idx}_${ed_idx}.entail.txt"


output_path="/home/Datasets/processed/Wiki10000/"
data_file_prefix="wiki10000"
st_idx=700000
ed_idx=750000
python3 run_glue.py \
        --model_type xlnet \
        --model_name_or_path xlnet-base-cased \
        --task_name MRPC \
        --do_test \
        --do_lower_case \
        --data_dir /home/Datasets/glue_data/MRPC/ \
        --max_seq_length 128 \
        --per_gpu_eval_batch_size=8   \
        --per_gpu_train_batch_size=8   \
        --learning_rate 2e-5 \
        --num_train_epochs 1.0 \
        --output_dir /home/Datasets/output/ET/xlnet-base-cased/ \
        --overwrite_output_dir \
        --context_question_answer_file "${output_path}${data_file_prefix}.qa.${st_idx}_${ed_idx}.uniq.postprocessed.txt" \
        --context_question_answer_columns 3 2 4 \
        --context_question_answer_score_file "${output_path}${data_file_prefix}.qa.${st_idx}_${ed_idx}.entail.txt"


output_path="/home/Datasets/processed/Wiki10000/"
data_file_prefix="wiki10000"
st_idx=750000
ed_idx=800000
python3 run_glue.py \
        --model_type xlnet \
        --model_name_or_path xlnet-base-cased \
        --task_name MRPC \
        --do_test \
        --do_lower_case \
        --data_dir /home/Datasets/glue_data/MRPC/ \
        --max_seq_length 128 \
        --per_gpu_eval_batch_size=8   \
        --per_gpu_train_batch_size=8   \
        --learning_rate 2e-5 \
        --num_train_epochs 1.0 \
        --output_dir /home/Datasets/output/ET/xlnet-base-cased/ \
        --overwrite_output_dir \
        --context_question_answer_file "${output_path}${data_file_prefix}.qa.${st_idx}_${ed_idx}.uniq.postprocessed.txt" \
        --context_question_answer_columns 3 2 4 \
        --context_question_answer_score_file "${output_path}${data_file_prefix}.qa.${st_idx}_${ed_idx}.entail.txt"


output_path="/home/Datasets/processed/Wiki10000/"
data_file_prefix="wiki10000"
st_idx=800000
ed_idx=850000
python3 run_glue.py \
        --model_type xlnet \
        --model_name_or_path xlnet-base-cased \
        --task_name MRPC \
        --do_test \
        --do_lower_case \
        --data_dir /home/Datasets/glue_data/MRPC/ \
        --max_seq_length 128 \
        --per_gpu_eval_batch_size=8   \
        --per_gpu_train_batch_size=8   \
        --learning_rate 2e-5 \
        --num_train_epochs 1.0 \
        --output_dir /home/Datasets/output/ET/xlnet-base-cased/ \
        --overwrite_output_dir \
        --context_question_answer_file "${output_path}${data_file_prefix}.qa.${st_idx}_${ed_idx}.uniq.postprocessed.txt" \
        --context_question_answer_columns 3 2 4 \
        --context_question_answer_score_file "${output_path}${data_file_prefix}.qa.${st_idx}_${ed_idx}.entail.txt"


output_path="/home/Datasets/processed/Wiki10000/"
data_file_prefix="wiki10000"
st_idx=850000
ed_idx=900000
python3 run_glue.py \
        --model_type xlnet \
        --model_name_or_path xlnet-base-cased \
        --task_name MRPC \
        --do_test \
        --do_lower_case \
        --data_dir /home/Datasets/glue_data/MRPC/ \
        --max_seq_length 128 \
        --per_gpu_eval_batch_size=8   \
        --per_gpu_train_batch_size=8   \
        --learning_rate 2e-5 \
        --num_train_epochs 1.0 \
        --output_dir /home/Datasets/output/ET/xlnet-base-cased/ \
        --overwrite_output_dir \
        --context_question_answer_file "${output_path}${data_file_prefix}.qa.${st_idx}_${ed_idx}.uniq.postprocessed.txt" \
        --context_question_answer_columns 3 2 4 \
        --context_question_answer_score_file "${output_path}${data_file_prefix}.qa.${st_idx}_${ed_idx}.entail.txt"


output_path="/home/Datasets/processed/Wiki10000/"
data_file_prefix="wiki10000"
st_idx=900000
ed_idx=950000
python3 run_glue.py \
        --model_type xlnet \
        --model_name_or_path xlnet-base-cased \
        --task_name MRPC \
        --do_test \
        --do_lower_case \
        --data_dir /home/Datasets/glue_data/MRPC/ \
        --max_seq_length 128 \
        --per_gpu_eval_batch_size=8   \
        --per_gpu_train_batch_size=8   \
        --learning_rate 2e-5 \
        --num_train_epochs 1.0 \
        --output_dir /home/Datasets/output/ET/xlnet-base-cased/ \
        --overwrite_output_dir \
        --context_question_answer_file "${output_path}${data_file_prefix}.qa.${st_idx}_${ed_idx}.uniq.postprocessed.txt" \
        --context_question_answer_columns 3 2 4 \
        --context_question_answer_score_file "${output_path}${data_file_prefix}.qa.${st_idx}_${ed_idx}.entail.txt"


output_path="/home/Datasets/processed/Wiki10000/"
data_file_prefix="wiki10000"
st_idx=950000
ed_idx=1000000
python3 run_glue.py \
        --model_type xlnet \
        --model_name_or_path xlnet-base-cased \
        --task_name MRPC \
        --do_test \
        --do_lower_case \
        --data_dir /home/Datasets/glue_data/MRPC/ \
        --max_seq_length 128 \
        --per_gpu_eval_batch_size=8   \
        --per_gpu_train_batch_size=8   \
        --learning_rate 2e-5 \
        --num_train_epochs 1.0 \
        --output_dir /home/Datasets/output/ET/xlnet-base-cased/ \
        --overwrite_output_dir \
        --context_question_answer_file "${output_path}${data_file_prefix}.qa.${st_idx}_${ed_idx}.uniq.postprocessed.txt" \
        --context_question_answer_columns 3 2 4 \
        --context_question_answer_score_file "${output_path}${data_file_prefix}.qa.${st_idx}_${ed_idx}.entail.txt"
