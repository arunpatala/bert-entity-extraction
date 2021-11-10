import config
import torch
import pandas as pd
from sklearn import preprocessing
import joblib 
from sklearn import model_selection

def process_data(data_path):
    df = pd.read_csv(data_path, encoding="latin-1")
    df.loc[:, "Sentence #"] = df["Sentence #"].fillna(method="ffill")

    enc_pos = preprocessing.LabelEncoder()
    enc_tag = preprocessing.LabelEncoder()

    df.loc[:, "POS"] = enc_pos.fit_transform(df["POS"])
    df.loc[:, "Tag"] = enc_tag.fit_transform(df["Tag"])

    sentences = df.groupby("Sentence #")["Word"].apply(list).values
    pos = df.groupby("Sentence #")["POS"].apply(list).values
    tag = df.groupby("Sentence #")["Tag"].apply(list).values
    return sentences, pos, tag, enc_pos, enc_tag

def get_lines(data_path):
  lines = [line.rstrip('\n') for line in open(data_path).readlines()]
  return lines

def get_sentences(lines):
  sentence = []
  for l in lines:
    if l=='':
      yield sentence
      sentence = []
    else:
      sentence.append(l.split("\t"))
  if len(sentence)!=0:
    yield sentence

def get_texts_tags(data_path):
  texts = []
  tags = []
  for s in get_sentences(get_lines(data_path)):
    text, tag = [i[0] for i in s], [i[1] for i in s]
    texts.append(text)
    tags.append(tag)
  enc = preprocessing.LabelEncoder()
  enc.fit([t for tag in tags for t in tag])
  return texts, tags, [enc.transform(tag) for tag in tags], enc

class BioEntityDataset:
    def __init__(self, texts, tags):
        self.texts = texts
        self.tags = tags
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, item):
        text = self.texts[item]
        tags = self.tags[item]

        ids = []
        target_tag =[]

        for i, s in enumerate(text):
            inputs = config.TOKENIZER.encode(
                s,
                add_special_tokens=False
            )
            input_len = len(inputs)
            ids.extend(inputs)
            target_tag.extend([tags[i]] * input_len)

        ids = ids[:config.MAX_LEN - 2]
        target_tag = target_tag[:config.MAX_LEN - 2]

        ids = [101] + ids + [102]
        target_tag = [0] + target_tag + [0]

        mask = [1] * len(ids)
        token_type_ids = [0] * len(ids)

        padding_len = config.MAX_LEN - len(ids)

        ids = ids + ([0] * padding_len)
        mask = mask + ([0] * padding_len)
        token_type_ids = token_type_ids + ([0] * padding_len)
        target_tag = target_tag + ([0] * padding_len)

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "target_tag": torch.tensor(target_tag, dtype=torch.long),
        }



if __name__ == "__main__":
    sentences, tags, tag_ids, enc_tag = get_texts_tags(config.BIO_TRAINING_FILE)

    le = enc_tag
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(le_name_mapping)

    print(abc)
    meta_data = {
        "enc_tag": enc_tag
    }

    joblib.dump(meta_data, "biometa.bin")

    num_tag = len(list(enc_tag.classes_))

    (
        train_sentences,
        test_sentences,
        train_tag,
        test_tag
    ) = model_selection.train_test_split(sentences, tag_ids, random_state=42, test_size=0.1)

    train_dataset = BioEntityDataset(
        texts=train_sentences, tags=train_tag
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=4
    )

    valid_dataset = BioEntityDataset(
        texts=test_sentences, tags=test_tag
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config.VALID_BATCH_SIZE, num_workers=1
    )

    for data in train_data_loader:
      print(list(data))
      for k in data:
        print(k, data[k].shape)
      print(abc)
