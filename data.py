import os
from functools import partial
from random import choice

from paddle.io import BatchSampler, DataLoader
from paddlenlp.data import Pad, Tuple
from paddlenlp.datasets import load_dataset

from utils import load_pickle, save_pickle
import collections

GLUE_PROCESSED = collections.OrderedDict(
    [

        (
            "xnli",
            (
                ["xnli hypothesis: ", " premise: "],
                ["contradictory", "entailment", "neutral"],
            ),
        ),
    ]
)



def trans_func(example, tokenizer, args):
    task_name = args.task_name
    processed, label = GLUE_PROCESSED[task_name]
    if label:
        id2label = dict(zip(range(len(label)), label))
    else:
        id2label = None

    if not args.is_test:
        if id2label:
            label_text = id2label[example["labels"]]
        else:
            label_text = str(example["labels"])
        target = tokenizer(
            label_text, return_token_type_ids=False, return_attention_mask=True
        )

    if len(processed) == 1:
        text = processed[0] + example["sentence"]
    else:
        text = processed[0] + example["sentence1"] + processed[1] + example["sentence2"]

    source = tokenizer(
        text,
        max_seq_len=args.max_seq_length,
        return_token_type_ids=False,
        return_attention_mask=True,
    )

    if not args.is_test:
        return (
            source["input_ids"],
            source["attention_mask"],
            target["input_ids"],
            target["attention_mask"],
        )
    else:
        return source["input_ids"], source["attention_mask"]

def read(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        # 跳过列名
        head = True
        for line in f:
            data = line.strip().split('\t')
            if not head:
                head = data
            else:
                labels = data[2]
                sentence1 = data[0]
                sentence2 = data[1]
                if labels == "contradictory":
                    labels = 0
                elif labels == "entailment":     
                    labels = 1
                elif labels == "neutral":     
                    labels = 2    
                yield {"sentence1": sentence1, "sentence2": sentence2, "labels": labels}

def get_train_dataloader_xnli(tokenizer, args):
    filename = os.path.join("work/data/", args.task_name + "_train" + ".pkl")

    if os.path.exists(filename):
        ds = load_pickle(filename)
    else:
        ds = load_dataset(read, data_path=f'data/train.en.tsv', lazy=False)
        ds.map(
            partial(trans_func, tokenizer=tokenizer, args=args),
            batched=False,
            lazy=False,
        )
        save_pickle(ds, filename)

    batch_sampler = BatchSampler(ds, batch_size=args.train_batch_size, shuffle=True)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id,dtype="int64"),  # input_ids
        Pad(axis=0, pad_val=tokenizer.pad_token_id,dtype="int64"),  # attention_mask
        Pad(axis=0, pad_val=-100,dtype="int64"),  # lm_labels
        Pad(axis=0, pad_val=tokenizer.pad_token_id,dtype="int64"),  # decoder_attention_mask
    ): fn(samples)

    data_loader = DataLoader(
        dataset=ds,
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        num_workers=args.num_workers,
        return_list=True,
    )

    return data_loader

def get_dev_dataloader_xnli(tokenizer, args):
    filename = os.path.join("work/data/", args.task_name + "_dev" + ".pkl")

    if os.path.exists(filename):
        ds = load_pickle(filename)
    else:
        ds = load_dataset(read, data_path=f'data/dev.en.tsv', lazy=False)
        ds.map(
            partial(trans_func, tokenizer=tokenizer, args=args),
            batched=False,
            lazy=False,
        )
        save_pickle(ds, filename)

    batch_sampler = BatchSampler(ds, batch_size=args.train_batch_size, shuffle=True)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id,dtype="int64"),  # input_ids
        Pad(axis=0, pad_val=tokenizer.pad_token_id,dtype="int64"),  # attention_mask
        Pad(axis=0, pad_val=-100,dtype="int64"),  # lm_labels
        Pad(axis=0, pad_val=tokenizer.pad_token_id,dtype="int64"),  # decoder_attention_mask
    ): fn(samples)

    data_loader = DataLoader(
        dataset=ds,
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        num_workers=args.num_workers,
        return_list=True,
    )

    return data_loader

def get_test_dataloader_xnli(tokenizer, args):
    filename = os.path.join("work/data/", args.task_name + "_test" + ".pkl")

    if os.path.exists(filename):
        ds = load_pickle(filename)
    else:
        ds = load_dataset(read, data_path=f'data/test.en.tsv', lazy=False)
        ds.map(
            partial(trans_func, tokenizer=tokenizer, args=args),
            batched=False,
            lazy=False,
        )
        save_pickle(ds, filename)

    batch_sampler = BatchSampler(ds, batch_size=args.train_batch_size, shuffle=True)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id,dtype="int64"),  # input_ids
        Pad(axis=0, pad_val=tokenizer.pad_token_id,dtype="int64"),  # attention_mask
        Pad(axis=0, pad_val=-100,dtype="int64"),  # lm_labels
        Pad(axis=0, pad_val=tokenizer.pad_token_id,dtype="int64"),  # decoder_attention_mask
    ): fn(samples)

    data_loader = DataLoader(
        dataset=ds,
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        num_workers=args.num_workers,
        return_list=True,
    )

    return data_loader

def get_train_dataloader(tokenizer, args):
    filename = os.path.join("caches", args.task_name + "_train" + ".pkl")

    if os.path.exists(filename):
        ds = load_pickle(filename)
    else:
        ds = load_dataset("glue", args.task_name, splits="train")
        ds.map(
            partial(trans_func, tokenizer=tokenizer, args=args),
            batched=False,
            lazy=False,
        )
        save_pickle(ds, filename)

    batch_sampler = BatchSampler(ds, batch_size=args.train_batch_size, shuffle=True)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id,dtype="int64"),  # input_ids
        Pad(axis=0, pad_val=tokenizer.pad_token_id,dtype="int64"),  # attention_mask
        Pad(axis=0, pad_val=-100,dtype="int64"),  # lm_labels
        Pad(axis=0, pad_val=tokenizer.pad_token_id,dtype="int64"),  # decoder_attention_mask
    ): fn(samples)

    data_loader = DataLoader(
        dataset=ds,
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        num_workers=args.num_workers,
        return_list=True,
    )

    return data_loader




def get_dev_dataloader(tokenizer, args):
    filename = os.path.join("caches", args.task_name + "_dev" + ".pkl")

    if os.path.exists(filename):
        ds = load_pickle(filename)
    else:
        ds = load_dataset("glue", args.task_name, splits="dev")
        ds.map(
            partial(trans_func, tokenizer=tokenizer, args=args),
            batched=False,
            lazy=False,
        )
        save_pickle(ds, filename)

    batch_sampler = BatchSampler(ds, batch_size=args.train_batch_size, shuffle=False)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id,dtype="int64"),  # input_ids
        Pad(axis=0, pad_val=tokenizer.pad_token_id,dtype="int64"),  # attention_mask
        Pad(axis=0, pad_val=-100,dtype="int64"),  # lm_labels
        Pad(axis=0, pad_val=tokenizer.pad_token_id,dtype="int64"),  # decoder_attention_mask
    ): fn(samples)

    data_loader = DataLoader(
        dataset=ds,
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        num_workers=args.num_workers,
        return_list=True,
    )

    return data_loader


def get_mnli_dev_dataloader(tokenizer, args, matched=True):
    if matched:
        split = "dev_matched"
    else:
        split = "dev_mismatched"
    filename = os.path.join("caches", args.task_name + f"_{split}" + ".pkl")
    if os.path.exists(filename):
        ds = load_pickle(filename)
    else:
        ds = load_dataset("glue", args.task_name, splits=split)
        ds.map(
            partial(trans_func, tokenizer=tokenizer, args=args),
            batched=False,
            lazy=False,
        )
        save_pickle(ds, filename)

    batch_sampler = BatchSampler(ds, batch_size=args.train_batch_size, shuffle=False)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id,dtype="int64"),  # input_ids
        Pad(axis=0, pad_val=tokenizer.pad_token_id,dtype="int64"),  # attention_mask
        Pad(axis=0, pad_val=-100,dtype="int64"),  # lm_labels
        Pad(axis=0, pad_val=tokenizer.pad_token_id,dtype="int64"),  # decoder_attention_mask
    ): fn(samples)

    data_loader = DataLoader(
        dataset=ds,
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        num_workers=args.num_workers,
        return_list=True,
    )

    return data_loader