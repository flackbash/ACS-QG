#!/bin/bash

bold=$(tput bold)
normal=$(tput sgr0)

# Get data directory from cmd line argument
if [ -z "$1" ]
then
      echo "${bold}Usage: $0 <data_dir>${normal}"
      exit 1
fi
# Make sure the data directory does not have a trailing /
datadir=${1%/}

# Check if requirements to run the script are installed.
echo "${bold}Checking requirements...${normal}"
if ! [ -x "$(command -v cabextract)" ]; then
  echo "${bold}Error: 'cabextract' is required for this script but is not installed or you do not have the necessary permissions.${normal}" >&2
  exit 1
fi

if ! [ -x "$(command -v python3)" ]; then
  echo "${bold}Error: 'python3' is required for this script but is not installed or you do not have the necessary permissions.${normal}" >&2
  exit 1
fi

if ! [ -x "$(command -v docker)" ]; then
  echo "${bold}Error: 'docker' is required for this script but is not installed or you do not have the necessary permissions.${normal}" >&2
  exit 1
fi

# Create directory structure in the data directory
echo "${bold}Creating directory structure...${normal}"
mkdir "$datadir"/original
mkdir "$datadir"/glue_data
mkdir "$datadir"/output
mkdir -p "$datadir"/processed/SQuAD1.1-Zhou
# Create directory structure in the data output directory
mkdir "$datadir"/output/checkpoint/
mkdir "$datadir"/output/figure/
mkdir "$datadir"/output/log/
mkdir "$datadir"/output/pkl/
mkdir "$datadir"/output/result/

# Adjust permissions of directories to make sure data can be written and accessed in a docker container
echo "${bold}Adjusting permissions...${normal}"
chmod a+rx "$datadir"/original
chmod a+rx "$datadir"/glue_data
chmod a+rwx "$datadir"/output
chmod a+rwx "$datadir"/output/*
chmod a+rwx "$datadir"/processed
chmod a+rwx "$datadir"/processed/*
# Adjust permissions of shell scripts to make them executable
chmod u+x ./*.sh


# Download datasets. Datasets are only downloaded if they do not exist yet in the target directory
# Set up Wiki10000 file
# Download Wiki10000.json from https://www.dropbox.com/s/mkwfazyr9bmrqc5/wiki10000.json.zip?dl=0
echo "${bold}Setting up Wiki10000 file...${normal}"
if [ ! -f "$datadir"/original/Wiki10000/wiki10000.json ]; then
  wget https://www.dropbox.com/s/mkwfazyr9bmrqc5/wiki10000.json.zip?dl=1 -O wiki10000.json.zip
  unzip wiki10000.json.zip -d "$datadir"/original/Wiki10000/
  rm -r "$datadir"/original/Wiki10000/__MACOSX/
  rm wiki10000.json.zip
  chmod a+rx "$datadir"/original/Wiki10000
  chmod a+r "$datadir"/original/Wiki10000/*
else
  echo "${bold}wiki10000.json already exists. Skip.${normal}"
fi


# Set up GloVe
# Download vectors wth 2.2M vocabulary from <https://nlp.stanford.edu/projects/glove/>
echo "${bold}Setting up GloVe...${normal}"
if [ ! -f "$datadir"/original/Glove/glove.840B.300d.txt ]; then
  wget nlp.stanford.edu/data/glove.840B.300d.zip
  unzip glove.840B.300d.zip -d "$datadir"/original/Glove/
  rm glove.840B.300d.zip
  chmod a+rx "$datadir"/original/Glove
  chmod a+r "$datadir"/original/Glove/*
else
  echo "${bold}glove.840B.300d.txt already exists. Skip.${normal}"
fi
if [ ! -f "$datadir"/original/Glove/glove.840B.300d.bin ]; then
  python3 create_glove_binary.py "$datadir"
  chmod a+r "$datadir"/original/Glove/*
else
  echo "${bold}glove.840B.300d.bin already exists. Skip.${normal}"
fi


# Set up BPEmbeddings
# Download embeddings from https://nlp.h-its.org/bpemb/
echo "${bold}Setting up BPEmbeddings...${normal}"
if [ ! -f "$datadir"/original/BPE/en.wiki.bpe.op50000.d100.w2v.txt ] || [ ! -f "$datadir"/original/BPE/en.wiki.bpe.op50000.model ]; then
  wget https://nlp.h-its.org/bpemb/en/en.wiki.bpe.op50000.d100.w2v.txt.tar.gz
  wget https://nlp.h-its.org/bpemb/en/en.wiki.bpe.op50000.model
  tar -xvzf en.wiki.bpe.op50000.d100.w2v.txt.tar.gz
  mkdir "$datadir"/original/BPE
  mv en.wiki.bpe.op50000.d100.w2v.txt "$datadir"/original/BPE/
  mv en.wiki.bpe.op50000.model "$datadir"/original/BPE/
  rm en.wiki.bpe.op50000.d100.w2v.txt.tar.gz
  chmod a+rx "$datadir"/original/BPE/
  chmod a+r "$datadir"/original/BPE/*
else
  echo "${bold}BPE files already exist. Skip.${normal}"
fi


# Set up Zhou-SQuAD1.1 data split
# Download data split from <https://res.qyzhou.me/redistribute.zip>
echo "${bold}Setting up Zhou-SQuAD1.1 data split...${normal}"
if [ ! -f "$datadir"/original/SQuAD1.1-Zhou/train.txt ] || [ ! -f "$datadir"/original/SQuAD1.1-Zhou/test.txt ] || [ ! -f "$datadir"/original/SQuAD1.1-Zhou/dev.txt ]; then
  wget https://res.qyzhou.me/redistribute.zip
  unzip redistribute.zip -d "$datadir"/original/SQuAD1.1-Zhou/
  mv "$datadir"/original/SQuAD1.1-Zhou/redistribute/raw/dev.txt.shuffle.dev "$datadir"/original/SQuAD1.1-Zhou/dev.txt
  mv "$datadir"/original/SQuAD1.1-Zhou/redistribute/raw/dev.txt.shuffle.test "$datadir"/original/SQuAD1.1-Zhou/test.txt
  mv "$datadir"/original/SQuAD1.1-Zhou/redistribute/raw/train.txt "$datadir"/original/SQuAD1.1-Zhou/train.txt
  rm -r "$datadir"/original/SQuAD1.1-Zhou/redistribute/
  rm -r redistribute.zip
  chmod a+rx "$datadir"/original/SQuAD1.1-Zhou/
  chmod a+r "$datadir"/original/SQuAD1.1-Zhou/*
else
  echo "${bold}Zhou-SQuAD1.1 files already exist. Skip.${normal}"
fi


# Set up SQuAD 2.0
# Download dataset from <https://rajpurkar.github.io/SQuAD-explorer/>
echo "${bold}Setting up SQuAD2.0 data...${normal}"
if [ ! -f "$datadir"/original/SQuAD2.0/train-v2.0.json ] || [ ! -f "$datadir"/original/SQuAD2.0/dev-v2.0.json ]; then
  wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json
  wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json
  mkdir "$datadir"/original/SQuAD2.0
  mv dev-v2.0.json "$datadir"/original/SQuAD2.0/
  mv train-v2.0.json "$datadir"/original/SQuAD2.0/
  chmod a+rx "$datadir"/original/SQuAD2.0/
  chmod a+r "$datadir"/original/SQuAD2.0/*
else
  echo "${bold}SQuAD2.0 files already exist. Skip.${normal}"
fi


# Move datasets contained in the Github repository to the appropriate location
echo "${bold}Setting up Github data...${normal}"
if [ ! -f "$datadir"/original/function-words/function_words.txt ] || [ ! -f "$datadir"/original/fixed-expressions/fixed_expressions.txt ]; then
  mkdir "$datadir"/original/function-words
  mkdir "$datadir"/original/fixed-expressions
  cp Datasets/function_words.txt "$datadir"/original/function-words/
  cp Datasets/fixed_expressions.txt "$datadir"/original/fixed-expressions/
  chmod a+rx "$datadir"/original/function-words
  chmod a+rx "$datadir"/original/fixed-expressions
else
  echo "${bold}Github files already at the right spot. Skip.${normal}"
fi


# Download MRPC
echo "${bold}Setting up MRPC data...${normal}"
if [ ! -f "$datadir"/glue_data/MRPC/train.tsv ] || [ ! -f "$datadir"/glue_data/MRPC/test.tsv ] || [ ! -f "$datadir"/glue_data/MRPC/dev.tsv ]; then
  wget https://download.microsoft.com/download/D/4/6/D46FF87A-F6B9-4252-AA8B-3604ED519838/MSRParaphraseCorpus.msi
  mkdir "$datadir"/glue_data/MRPC
  cabextract MSRParaphraseCorpus.msi -d "$datadir"/glue_data/MRPC
  tr -d $'\r' < "$datadir"/glue_data/MRPC/_2DEC3DBE877E4DB192D17C0256E90F1D > "$datadir"/glue_data/MRPC/train.tsv
  tr -d $'\r' < "$datadir"/glue_data/MRPC/_D7B391F9EAFF4B1B8BCE8F21B20B1B61 > "$datadir"/glue_data/MRPC/test_tmp.tsv
  # I couldn't find dev.tsv anywhere, so instead I split the test set into two parts, test.tsv and dev.tsv.
  # This is most likely not the original split though
  head -n 1000 "$datadir"/glue_data/MRPC/test_tmp.tsv > "$datadir"/glue_data/MRPC/test.tsv
  head -n 1 "$datadir"/glue_data/MRPC/test_tmp.tsv > "$datadir"/glue_data/MRPC/dev.tsv
  tail -n +1001 "$datadir"/glue_data/MRPC/test_tmp.tsv >> "$datadir"/glue_data/MRPC/dev.tsv
  rm "$datadir"/glue_data/MRPC/_*
  rm "$datadir"/glue_data/MRPC/test_tmp.tsv
  rm MSRParaphraseCorpus.msi
  chmod a+rx "$datadir"/glue_data/MRPC/
  chmod a+r "$datadir"/glue_data/MRPC/*
else
  echo "${bold}MRPC files already exist. Skip.${normal}"
fi


# Build docker container
echo "${bold}Building docker image...${normal}"
docker build -t acs-qg-docker .

# Create script to start a docker container
echo "${bold}Creating docker start script...${normal}"
touch start_docker_container.sh
chmod u+x start_docker_container.sh
echo "docker run -it -v ${datadir}:/home/Datasets -v ${datadir}/output:/home/FQG/output acs-qg-docker" > start_docker_container.sh

echo "${bold}All done. You can start the docker container by running ./start_docker_container.sh${normal}"
