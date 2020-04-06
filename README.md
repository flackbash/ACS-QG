# ACS-QG
Factorized question generation for controlled MRC training data generation.

This is an unofficial, self-contained version of the
[code](https://github.com/BangLiu/ACS-QG) for the
[paper](https://arxiv.org/pdf/2002.00748.pdf)
*"Asking Questions the Human Way: Scalable Question-Answer Generation from Text Corpus"*.
Please cite this paper if it is useful for your projects.

This code does not come with any warranties.
Changes to the original code were kept to a minimum and mainly directed at making the code self-contained
and compatible with the corresponding Dockerfile and Polyaxon.
Other changes deemed notable are listed in the corresponding subsection of this README.

## Setup
The following script takes care of downloading all necessary files as well as setting up the required directory
structure so you don't need to worry about a thing. Simply create a directory `<data_directory>` in which all
ACS-QG-related data will be stored and run

    ./setup.sh <data_directory>

This will also build a docker image and create a script `start_docker_container.sh` which will start a docker
container with the appropriate parameters.

In the docker container, you can now run the experiments as given in the
[original code](https://github.com/BangLiu/ACS-QG).

## Run experiments
Within the docker container you can run the following experiments:

1. **Check your environment**\
    Run

       ./experiments_0_debug.sh

    It should run through without errors.
    If it does it means your environment is set up correctly.
    
2. **Train models**\
    Run the following scripts in parallel on different GPUs to save time

       ./experiments_1_ET_train.sh  # runs run_glue.py
       ./experiments_1_QG_train_gpt2.sh  # runs QG_gpt2_train.py
       ./experiments_1_QG_train_seq2seq.sh  # runs QG_main.py
    
3. **Perform data augmentation**\
    Sequentially run
 
       ./experiments_2-DA_file2sents.sh  # runs DA_main.py
       ./experiments_3_DA_sents2augsents.sh  # runs DA_main.py
    
4. **Generate questions**\
    Generate questions using the GPT2 model or the seq2seq model. The experiments can be run in parallel.\
    **Note:** This step requires a GPU to efficiently process the data.
    
       ./experiments_4_QG_generate_gpt2.sh  # runs QG_gpt2_generate.py
       ./experiments_4_QG_generate_seq2seq.sh  # runs QG_augment_main.py

5. **Remove duplicate data**\
   To remove duplicate data run
   
       ./experiments_5_uniq_seq2seq.sh

6. **Post-process data**\
    Post-process seq2seq results to handle the repeat problem. It is not required if you use gpt2.

       ./experiments_6_postprocess_seq2seq.sh

*TODO: check how Data Evaluation (DE) to filter low-quality data samples fits in here*

## Run experiments with Polyaxon
If you want to use [Polyaxon](https://polyaxon.com/) to run your experiments,
you can do so using the scripts and commands described below.\
After an experiment has completed, download the output files and move them to the appropriate directory.
Refer to the *Input & output* section to find the appropriate directory.\
You might also need to change the `DATA_PATH` in `common/constants.py` such that it points to a valid Polyaxon
data directory.

2. **Train models**\
To train the models add the following commands to `polyaxonfile.yaml`,
where `<data_directory>` is the directory in which all your ACS-QG-related data is stored:

       python3 experiments_1_ET_train_poly.py <data_directory>  # tested
       python3 experiments_1_QG_train_gpt2_poly.py <data_directory>
       python3 experiments_1_QG_train_seq2seq_poly.py

4. **Generate questions**\
To generate questions using the models add the following commands to `polyaxonfile.yaml`:

       python3 experiments_4_QG_generate_seq2seq_poly.py <data_directory>
       python3 experiments_4_QG_generate_gpt2_poly.py <data_directory>


*TODO: add more Polyaxon commands*


## Inputs & outputs
This section lists for each expeciment the input files that will be read and the output files that will be
created.
The lists might not be exhaustive.

####`experiments_1_ET_train.sh`
Reads:

      <data_directory>/glue_data/MRPC/  # --data_dir

Writes:

      <data_directory>/output/ET/xlnet-base-cased/added_tokens.json  # --output_dir
      <data_directory>/output/ET/xlnet-base-cased/config.json  # --output_dir
      <data_directory>/output/ET/xlnet-base-cased/eval_results.txt  # --output_dir
      <data_directory>/output/ET/xlnet-base-cased/pytorch_model.bin  # --output_dir
      <data_directory>/output/ET/xlnet-base-cased/special_tokens_map.json  # --output_dir
      <data_directory>/output/ET/xlnet-base-cased/spiece.model  # --output_dir
      <data_directory>/output/ET/xlnet-base-cased/test_results.txt  # --output_dir
      <data_directory>/output/ET/xlnet-base-cased/training_args.bin  # --output_dir
      <data_directory>/output/ET/xlnet-base-cased/checkpoint-100/config.json  # --output_dir
      <data_directory>/output/ET/xlnet-base-cased/checkpoint-100/pytorch_model.bin  # --output_dir
      <data_directory>/output/ET/xlnet-base-cased/checkpoint-100/training_args.bin  # --output_dir
      <data_directory>/output/ET/xlnet-base-cased/checkpoint-200/[same as for checkpoint-100]  # --output_dir
      <data_directory>/output/ET/xlnet-base-cased/checkpoint-300/[same as for checkpoint-100]  # --output_dir
      <data_directory>/output/ET/xlnet-base-cased/checkpoint-400/[same as for checkpoint-100]  # --output_dir
      <data_directory>/output/ET/xlnet-base-cased/checkpoint-500/[same as for checkpoint-100]  # --output_dir

####`experiments_1_QG_train_gpt2.sh`
Reads:

      <data_directory>/original/SQuAD1.1-Zhou/train.txt  # --train_dataset_path
      <data_directory>/original/SQuAD1.1-Zhou/dev.txt  # --dev_dataset_path
      
Writes:

      <data_directory>/output/QG/gpt2_question_generation/[...]  # --output_dir

####`experiments_1_QG_train_seq2seq.sh`
Reads:

      ?

Writes:

      <data_directory>/output/checkpoint/FQG_squad_hard-oov_1_128_False_False_True_True_False_False_True_20/FQG_checkpoint_epoch<x>....pth.tar  # --checkpoint_dir
      <data_directory>/output/checkpoint/FQG_squad_hard-oov_1_128_False_False_True_True_False_False_True_20/model_best.pth.tar  # --checkpoint_dir
      <data_directory>/processed/SQuAD1.1-Zhou/counters.pkl  # --counters_file
      <data_directory>/processed/SQuAD1.1-Zhou/dev-eval.pkl  # --dev_eval_file
      <data_directory>/processed/SQuAD1.1-Zhou/dev-examples.pkl  # --dev_examples_file
      <data_directory>/processed/SQuAD1.1-Zhou/dev-meta.pkl  # --dev_meta_file
      <data_directory>/processed/SQuAD1.1-Zhou/emb_dicts.pkl  # --emb_dicts_file
      <data_directory>/processed/SQuAD1.1-Zhou/emb_mats.pkl  # --emb_mats_file
      <data_directory>/processed/SQuAD1.1-Zhou/related_words_dict.pkl  # --related_words_dict_file
      <data_directory>/processed/SQuAD1.1-Zhou/related_words_ids_mat.pkl  # --related_words_ids_mat_file
      <data_directory>/processed/SQuAD1.1-Zhou/test-eval.pkl  # --test_eval_file
      <data_directory>/processed/SQuAD1.1-Zhou/test-examples.pkl  # --test_examples_file
      <data_directory>/processed/SQuAD1.1-Zhou/test-meta.pkl  # --test_meta_file
      <data_directory>/processed/SQuAD1.1-Zhou/train-eval.pkl  # --train_eval_file
      <data_directory>/processed/SQuAD1.1-Zhou/train-examples.pkl  # --train_examples_file
      <data_directory>/processed/SQuAD1.1-Zhou/train-meta.pkl  # --train_meta_file

####`experiments_2-DA_file2sents.sh`
Reads:

      <data_directory>/original/Wiki10000/SQuAD2.0/train-v2.0.json  # --da_input_file
      <data_directory>/original/Wiki10000/wiki10000.json  # --da_input_file
      
Writes:

    <data_directory>/processed/SQuAD2.0/train.sentences.txt  # --da_sentences_file
    <data_directory>/processed/SQuAD2.0/train.paragraphs.txt  # --da_paragraphs_file
    <data_directory>/processed/Wiki10000/wiki10000.sentences.txt  # --da_sentences_file
    <data_directory>/processed/Wiki10000/wiki10000.paragraphs.txt  # --da_paragraphs_file

####`experiments_3_DA_sents2augsents.sh`
Reads:

    <data_directory>/original/SQuAD2.0/train-v2.0.json  # --da_input_file
    <data_directory>/original/Wiki10000/wiki10000.json  # --da_input_file

Writes:

    <data_directory>/processed/SQuAD2.0/train.sentences.txt  # --da_sentences_file
    <data_directory>/processed/SQuAD2.0/train.paragraphs.txt  # --da_paragraphs_file
    <data_directory>/processed/SQuAD2.0/train.sentences.augmented.<x>_<y>.pkl  # --da_augmented_sentences_file
    <data_directory>/processed/Wiki10000/wiki10000.sentences.txt  # --da_sentences_file
    <data_directory>/processed/Wiki10000/wiki10000.paragraphs.txt  # --da_paragraphs_file
    <data_directory>/processed/Wiki10000/wiki10000.sentences.augmented.<x>_<y>.pkl  # --da_augmented_sentences_file

####`experiments_4_QG_generate_gpt2.sh`
Reads:

    <data_directory>/output/QG/gpt2_question_generation/4epochs/2batchsize/[...]  # --model_name_or_path
    <data_directory>/processed/SQuAD2.0/train.sentences.augmented.<x>_<y>.pkl  # --filename
    <data_directory>/processed/Wiki10000/wiki10000.sentences.augmented.<x>_<y>.pkl  # --filename

Writes:

    <data_directory>/processed/SQuAD2.0/train.sentences.augmented.<x>_<y>.cache.qg.gpt2.pth  # --filecache
    <data_directory>/processed/SQuAD2.0/train.qa.<x>_<y>.qg.generated.gpt2.json  # --output_file
    <data_directory>/processed/Wiki10000/wiki10000.sentences.augmented.<x>_<y>.cache.qg.gpt2.pth  # --filecache
    <data_directory>/processed/Wiki10000/wiki10000.qa.<x>_<y>.qg.generated.gpt2.json  # --output_file

####`experiments_4_QG_generate_seq2seq.sh`
Reads:

    <data_directory>/output/checkpoint/  # --checkpoint_dir
    <data_directory>/processed/SQuAD2.0/train.paragraphs.txt  # --da_paragraphs_file
    <data_directory>/processed/Wiki10000/wiki10000.paragraphs.txt  # --da_paragraphs_file
    <data_directory>/processed/SQuAD2.0/train.sentences.augmented.<x>_<y>.pkl  # --da_augmented_sentences_file
    <data_directory>/processed/Wiki10000/wiki10000.sentences.augmented.<x>_<y>.pkl  # --da_augmented_sentences_file      

Writes:

    <data_directory>/processed/SQuAD2.0/train.sentences.augmented.<x>_<y>.processed.pkl  #  --qg_augmented_sentences_file
    <data_directory>/processed/Wiki10000/wiki10000.sentences.augmented.<x>_<y>.processed.pkl  #  --qg_augmented_sentences_file
    <data_directory>/processed/SQuAD2.0/train.sentences.augmented.<x>_<y>.output.txt  #  --qg_result_file
    <data_directory>/processed/Wiki10000/wiki10000.sentences.augmented.<x>_<y>.output.txt  #  --qg_result_file
    <data_directory>/processed/SQuAD2.0/train.qa.<x>_<y>.txt  #  --qa_data_file
    <data_directory>/processed/Wiki10000/wiki10000.qa.<x>_<y>.txt  #  --qa_data_file

####`experiments_5_uniq_seq2seq.sh`
Reads:

    <data_directory>/processed/SQuAD2.0/train.qa.<x>_<y>.txt
    <data_directory>/processed/Wiki10000/wiki10000.qa.<x>_<y>.txt

Writes:

    <data_directory>/processed/SQuAD2.0/train.qa.<x>_<y>.uniq.txt
    <data_directory>/processed/Wiki10000/wiki10000.qa.<x>_<y>.uniq.txt

####`experiments_6_postprocess_seq2seq.sh`
Reads:

    <data_directory>/processed/SQuAD2.0/train.qa.<x>_<y>.uniq.txt  # --input_file
    <data_directory>/processed/Wiki10000/wiki10000.qa.<x>_<y>.uniq.txt  # --input_file
Writes:

    <data_directory>/processed/SQuAD2.0/train.qa.<x>_<y>.uniq.postprocessed.txt  # --output_file
    <data_directory>/processed/Wiki10000/wiki10000.qa.<x>_<y>.uniq.postprocessed.txt  # --output_file


## Notable changes to the original code
* I couldn't find `dev.tsv` for the MRPC dataset anywhere, so instead I split the test set into two parts,
`test.tsv` and `dev.tsv`. This is most likely not the original split.