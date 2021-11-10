import transformers

MAX_LEN = 128
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 8
EPOCHS = 10
BASE_MODEL_PATH = "dmis-lab/biobert-base-cased-v1.2"
MODEL_PATH = "model_biobert.bin"
TRAINING_FILE = "../input/ner_dataset.csv"
BIO_TRAINING_FILE = "train_dev.tsv"
BIO_TESTING_FILE = "test.tsv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    do_lower_case=False
)
