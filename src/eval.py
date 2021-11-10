import numpy as np

import joblib
import torch

import config
import dataset
import engine
from model import EntityModel
from train import process_data
from sklearn import model_selection
from tqdm import tqdm 

if __name__ == "__main__":
  
    sentences, pos, tag, enc_pos, enc_tag = process_data(config.TRAINING_FILE)

    meta_data = joblib.load("meta.bin")
    enc_pos = meta_data["enc_pos"]
    enc_tag = meta_data["enc_tag"]

    num_pos = len(list(enc_pos.classes_))
    num_tag = len(list(enc_tag.classes_))

    (
        train_sentences,
        test_sentences,
        train_pos,
        test_pos,
        train_tag,
        test_tag
    ) = model_selection.train_test_split(sentences, pos, tag, random_state=42, test_size=0.1)

    train_dataset = dataset.EntityDataset(
        texts=train_sentences, pos=train_pos, tags=train_tag
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=4
    )

    valid_dataset = dataset.EntityDataset(
        texts=test_sentences, pos=test_pos, tags=test_tag
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config.VALID_BATCH_SIZE, num_workers=1
    )


    device = torch.device("cuda")
    model = EntityModel(num_tag=num_tag, num_pos=num_pos)
    model.load_state_dict(torch.load(config.MODEL_PATH))
    model.to(device)
    cnt1 = 0
    cnt_all1 = 0
    cnt2 = 0
    cnt_all2 = 0
    for data in tqdm(valid_data_loader):
        for k, v in data.items():
            data[k] = v.to(device)
        tag, pos, _ = model(**data)
        
        
        tag = tag.argmax(2).cpu()
        target_tag = data['target_tag'].cpu()
    
        mask = data['mask'].cpu()
        cnt1 += ((tag==target_tag)*mask).sum().item()
        cnt_all1 += mask.sum().item()
        cnt2 += ((tag==16)*mask).sum().item()
        cnt_all2 += mask.sum().item()
    print( "1", (cnt1*100.0)/cnt_all1)
    print( "2", (cnt2*100.0)/cnt_all2)

        


    sentence = """
    abhishek is going to india
    """
    tokenized_sentence = config.TOKENIZER.encode(sentence)

    sentence = sentence.split()
    print(sentence)
    print(tokenized_sentence)
    input_tokens = config.TOKENIZER.convert_ids_to_tokens(tokenized_sentence)
    print(input_tokens)

    test_dataset = dataset.EntityDataset(
        texts=[sentence], 
        pos=[[0] * len(sentence)], 
        tags=[[0] * len(sentence)]
    )


    with torch.no_grad():
        data = test_dataset[0]
        for k, v in data.items():
            data[k] = v.to(device).unsqueeze(0)
        tag, pos, _ = model(**data)

        print(
            enc_tag.inverse_transform(
                tag.argmax(2).cpu().numpy().reshape(-1)
            )[:len(tokenized_sentence)]
        )
        print(
            enc_pos.inverse_transform(
                pos.argmax(2).cpu().numpy().reshape(-1)
            )[:len(tokenized_sentence)]
        )


        
        output_tags = enc_tag.inverse_transform(tag.argmax(2).cpu().numpy().reshape(-1))[:len(tokenized_sentence)]
        print(list(zip(input_tokens, output_tags)))
