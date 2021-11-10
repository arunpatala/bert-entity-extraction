import torch
from tqdm import tqdm


def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()
    final_loss = 0
    cnt = 0
    correct_cnt = 0
    pbar = tqdm(data_loader, total=len(data_loader))
    for data in pbar:
        for k, v in data.items():
            data[k] = v.to(device)
        optimizer.zero_grad()
        tag, pos, loss = model(**data)
        loss.backward()
        optimizer.step()
        scheduler.step()
        final_loss += loss.item()
        correct_cnt += ((tag.argmax(2)==data["target_tag"])*data["mask"]).sum().item()
        cnt += data["mask"].sum().item()
        acc = (correct_cnt *100.0)/cnt
        pbar.set_description("acc: {}".format(acc))
        
    print("train acc", acc)
    return final_loss / len(data_loader)


def eval_fn(data_loader, model, device):
    model.eval()
    final_loss = 0
    cnt = 0
    correct_cnt = 0
    pbar = tqdm(data_loader, total=len(data_loader))
    for data in pbar:
        for k, v in data.items():
            data[k] = v.to(device)
        tag, pos, loss = model(**data)
        final_loss += loss.item()
        correct_cnt += ((tag.argmax(2)==data["target_tag"])*data["mask"]).sum().item()
        cnt += data["mask"].sum().item()
        acc = (correct_cnt *100.0)/cnt
        pbar.set_description("acc: {}".format(acc))
        
    print("test acc", acc)
    return final_loss / len(data_loader)
