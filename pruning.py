import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch_pruning as tp
from torch.utils.data.dataloader import DataLoader
from vietocr.loader.aug import ImgAugTransformV2
from vietocr.loader.dataloader import ClusterRandomSampler, Collator, OCRDataset
from vietocr.optim.labelsmoothingloss import LabelSmoothingLoss
from vietocr.tool.config import Cfg
from vietocr.tool.translate import batch_translate_beam_search, build_model, translate
from vietocr.tool.utils import compute_accuracy


class CosineAnnealingLR(object):
    def __init__(self, epochs, step_each_epoch, warmup_epoch=0, last_epoch=-1, **kwargs):
        super(CosineAnnealingLR, self).__init__()
        self.epochs = epochs * step_each_epoch
        self.last_epoch = last_epoch
        self.warmup_epoch = warmup_epoch * step_each_epoch

    def __call__(self, optimizer):
        return optim.lr_scheduler.LambdaLR(optimizer, self.lambda_func, self.last_epoch)

    def lambda_func(self, current_step, num_cycles=0.5):
        if current_step < self.warmup_epoch:
            return float(current_step) / float(max(1, self.warmup_epoch))
        progress = float(current_step - self.warmup_epoch) / float(max(1, self.epochs - self.warmup_epoch))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))


def load_dataloader(cfg, vocab, transform=ImgAugTransformV2()):
    dataset_name = cfg["dataset"].pop("name")
    data_root = cfg["dataset"].pop("data_root")
    train_anno, valid_anno = cfg["dataset"].pop("train_annotation"), cfg["dataset"].pop("valid_annotation")

    train_set = OCRDataset(
        lmdb_path=f"train_{dataset_name}",
        root_dir=data_root,
        annotation_path=train_anno,
        transform=transform,
        vocab=vocab,
        **cfg["dataset"],
    )
    valid_set = OCRDataset(
        lmdb_path=f"valid_{dataset_name}",
        root_dir=data_root,
        annotation_path=valid_anno,
        transform=transform,
        vocab=vocab,
        **cfg["dataset"],
    )

    batch_size = cfg["dataloader"].pop("batch_size")

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        sampler=ClusterRandomSampler(data_source=train_set, batch_size=batch_size, shuffle=True),
        collate_fn=Collator(masked_language_model=True),
        shuffle=False,
        drop_last=False,
        **cfg["dataloader"],
    )
    valid_loader = DataLoader(
        dataset=valid_set,
        batch_size=batch_size,
        sampler=ClusterRandomSampler(data_source=valid_set, batch_size=batch_size, shuffle=True),
        collate_fn=Collator(masked_language_model=False),
        shuffle=False,
        drop_last=False,
        **cfg["dataloader"],
    )

    return train_loader, valid_loader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="./config/vgg-transformer-slim.yml")
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    config = Cfg.load_config_from_file(args.config_path)
    model, vocab = build_model(config=config)

    device = config.pop("device", None) or "cpu"

    pretrained_weight = torch.load(config["pretrain"], map_location=device)
    model.load_state_dict(state_dict=pretrained_weight, strict=True)

    img = torch.randn(32, 3, 32, 300).to(device)
    tgt_input = torch.randint(0, len(vocab), (93, 32)).to(device)
    tgt_padding_mask = torch.BoolTensor(torch.randn(32, 93) < 0.05).to(device)
    example_inputs = (img, tgt_input, tgt_padding_mask)

    num_heads = {}
    ignored_layers = []
    channel_groups = {}
    unwrapped_parameters = None

    for m in model.modules():
        if isinstance(m, nn.Linear) and m.out_features == model:
            ignored_layers.append(m)

        if isinstance(m, nn.MultiheadAttention):
            channel_groups[m] = m.num_heads
            num_heads[m] = m.num_heads

    slim_mode = config["slim"].pop("mode")

    imp = tp.importance.GroupHessianImportance()
    pruner = tp.pruner.MetaPruner(
        model,
        example_inputs,
        ignored_layers=ignored_layers,
        unwrapped_parameters=unwrapped_parameters,
        channel_groups=channel_groups,
        num_heads=num_heads,
        importance=imp,
        **config["slim"],
    )

    model.zero_grad()
    imp.zero_grad()

    train_loader, valid_loader = load_dataloader(cfg=config, vocab=vocab)

    total_loss = 0
    train_losses = []

    total_loader_time = 0
    total_gpu_time = 0
    best_acc = 0

    num_iters = config["trainer"]["num_iters"]
    print_every = config["trainer"]["print_every"]
    valid_every = config["trainer"]["valid_every"]
    export_weight = config["trainer"]["export"]

    beamsearch = config["predictor"]["beamsearch"]

    data_iter = iter(train_loader)

    optimizer = optim.AdamW(model.parameters(), betas=(0.9, 0.98), eps=1e-09)
    lr_scheduler = CosineAnnealingLR(
        epochs=num_iters // len(train_loader),
        step_each_epoch=len(train_loader),
        **config["lr_scheduler"],
    )(optimizer=optimizer)

    criterion = LabelSmoothingLoss(len(vocab), padding_idx=vocab.pad, smoothing=0.1)

    for it in range(num_iters):
        start = time.time()

        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        total_loader_time += time.time() - start

        start = time.time()
        loss = step(model, batch, criterion, optimizer, device)
        total_gpu_time += time.time() - start

        total_loss += loss
        train_losses.append((it, loss))

        imp.accumulate_grad(model)

        if it % print_every == 0:
            info = "iter: {:06d} - train loss: {:.3f} - lr: {:.2e} - load time: {:.2f} - gpu time: {:.2f}".format(
                it,
                total_loss / print_every,
                optimizer.param_groups[0]["lr"],
                total_loader_time,
                total_gpu_time,
            )

            total_loss = 0
            total_loader_time = 0
            total_gpu_time = 0
            print(info)

        if it % valid_every == 0:
            val_loss = validate(model, valid_loader, criterion, device)
            acc_full_seq, acc_per_char = precision(model, valid_loader, beamsearch, vocab, device)

            info = "iter: {:06d} - valid loss: {:.3f} - acc full seq: {:.4f} - acc per char: {:.4f}".format(
                it, val_loss, acc_full_seq, acc_per_char
            )
            print(info)

            if acc_full_seq > best_acc:
                path, _ = os.path.split(export_weight)
                os.makedirs(path, exist_ok=True)

                torch.save(model.state_dict(), export_weight)

                best_acc = acc_full_seq

        if it % len(train_loader) == 0:
            lr_scheduler.step()

    for i, g in enumerate(pruner.step(interactive=True)):
        g.prune()


def step(model, batch, criterion, optimizer, device="cpu"):
    batch = batch_to_device(batch, device)
    img, tgt_input, tgt_output, tgt_padding_mask = (
        batch["img"],
        batch["tgt_input"],
        batch["tgt_output"],
        batch["tgt_padding_mask"],
    )

    outputs = model(img, tgt_input, tgt_key_padding_mask=tgt_padding_mask)
    outputs = outputs.view(-1, outputs.size(2))  # flatten(0, 1)
    tgt_output = tgt_output.view(-1)  # flatten()

    loss = criterion(outputs, tgt_output)

    optimizer.zero_grad()

    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

    optimizer.step()

    loss_item = loss.item()

    return loss_item


def validate(model, valid_loader, criterion, device="cpu"):
    model.eval()

    total_loss = []

    with torch.no_grad():
        for step, batch in enumerate(valid_loader):
            batch = batch_to_device(batch, device)
            img, tgt_input, tgt_output, tgt_padding_mask = (
                batch["img"],
                batch["tgt_input"],
                batch["tgt_output"],
                batch["tgt_padding_mask"],
            )

            outputs = model(img, tgt_input, tgt_padding_mask)
            outputs = outputs.flatten(0, 1)

            tgt_output = tgt_output.flatten()
            loss = criterion(outputs, tgt_output)

            total_loss.append(loss.item())

            del outputs
            del loss

    total_loss = np.mean(total_loss)

    model.train()

    return total_loss


def batch_to_device(batch, device="cpu"):
    img = batch["img"].to(device, non_blocking=True)
    tgt_input = batch["tgt_input"].to(device, non_blocking=True)
    tgt_output = batch["tgt_output"].to(device, non_blocking=True)
    tgt_padding_mask = batch["tgt_padding_mask"].to(device, non_blocking=True)

    batch = {
        "img": img,
        "tgt_input": tgt_input,
        "tgt_output": tgt_output,
        "tgt_padding_mask": tgt_padding_mask,
        "filenames": batch["filenames"],
    }

    return batch


def predict(model, valid_loader, beamsearch, vocab, device="cpu"):
    pred_sents = []
    actual_sents = []
    img_files = []

    for batch in valid_loader:
        batch = batch_to_device(batch, device)

        if beamsearch:
            translated_sentence = batch_translate_beam_search(batch["img"], model)
            prob = None
        else:
            translated_sentence, prob = translate(batch["img"], model)

        pred_sent = vocab.batch_decode(translated_sentence.tolist())
        actual_sent = vocab.batch_decode(batch["tgt_output"].tolist())

        img_files.extend(batch["filenames"])

        pred_sents.extend(pred_sent)
        actual_sents.extend(actual_sent)

    return pred_sents, actual_sents, img_files, prob


def precision(model, valid_loader, beamsearch, vocab, device="cpu"):
    pred_sents, actual_sents, _, _ = predict(model, valid_loader, beamsearch, vocab, device)

    acc_full_seq = compute_accuracy(actual_sents, pred_sents, mode="full_sequence")
    acc_per_char = compute_accuracy(actual_sents, pred_sents, mode="per_char")

    return acc_full_seq, acc_per_char


if __name__ == "__main__":
    main()
