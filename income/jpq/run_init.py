# coding=utf-8
'''
Encoding part is modified base on 
    https://github.com/jingtaozhan/DRhard/blob/main/star/inference.py (SIGIR'21)
'''
import os
import torch
import faiss
import argparse
import subprocess
import logging
import numpy as np
from tqdm import tqdm
from transformers import RobertaConfig, AutoConfig, AutoModel
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SequentialSampler

from jpq.model import TASBDot
from jpq.dataset import (
    TextTokenIdsCache, SequenceDataset,
    get_collate_function
)

logger = logging.getLogger(__name__)


def prediction(
    model, data_collator, device, output_dir, eval_batch_size, 
    n_gpu, test_dataset, embedding_memmap
):
    os.makedirs(output_dir, exist_ok=True)
    test_dataloader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=eval_batch_size*n_gpu,
        collate_fn=data_collator,
        drop_last=False,
    )
    # multi-gpu eval
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    batch_size = test_dataloader.batch_size
    num_examples = len(test_dataloader.dataset)
    logger.info("***** Running *****")
    logger.info("  Num examples = %d", num_examples)
    logger.info("  Batch size = %d", batch_size)

    model.eval()
    write_index = 0
    for step, (inputs, ids) in enumerate(tqdm(test_dataloader)):
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(device)
        with torch.no_grad():
            logits = model(**inputs).detach().cpu().numpy()
        write_size = len(logits)
        assert write_size == len(ids)
        embedding_memmap[write_index:write_index+write_size] = logits
        write_index += write_size
    assert write_index == len(embedding_memmap)


def doc_inference(
    model, device, max_doc_length, output_dir, eval_batch_size, n_gpu, 
                  preprocess_dir, doc_embed_path, embedding_size
):
    doc_collator = get_collate_function(max_doc_length)
    ids_cache = TextTokenIdsCache(data_dir=preprocess_dir, prefix="passages")
    doc_dataset = SequenceDataset(
        ids_cache=ids_cache,
        max_seq_length=max_doc_length
    )
    assert not os.path.exists(doc_embed_path)
    doc_memmap = np.memmap(doc_embed_path, 
        dtype=np.float32, mode="w+", shape=(len(doc_dataset), embedding_size))
    try:
        prediction(model, doc_collator, device, output_dir, eval_batch_size, n_gpu,
            doc_dataset, doc_memmap)
    except:
        subprocess.check_call(["rm", doc_embed_path])
        raise

def init(
    preprocess_dir: str,
    model_dir: str,
    output_dir: str,
    subvector_num: int,
    max_doc_length: int = 512,
    eval_batch_size: int = 128,
    doc_embed_size: int = 768,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    logger.info("Model Device: %s, n_gpu: %s", device, n_gpu)

    os.makedirs(output_dir, exist_ok=True)
    doc_embed_path = os.path.join(output_dir, "doc_embed.memmap")

    if not os.path.exists(doc_embed_path):
        logger.info("Encoding passages to dense vectors ...")
        config = AutoConfig.from_pretrained(model_dir, gradient_checkpointing=False)
        model = TASBDot.from_pretrained(model_dir, config=config)
        model = model.to(device)
        doc_inference(model, device, max_doc_length, output_dir, eval_batch_size, n_gpu, 
                  preprocess_dir, doc_embed_path, doc_embed_size)
        model = None
        torch.cuda.empty_cache()
    else:
        logger.info(f"{doc_embed_path} exists, skip encoding procedure")

    doc_embeddings = np.memmap(doc_embed_path, 
        dtype=np.float32, mode="r")
    doc_embeddings = doc_embeddings.reshape(-1, doc_embed_size)

    save_index_path = os.path.join(output_dir, f"OPQ{subvector_num},IVF1,PQ{subvector_num}x8.index")
    res = faiss.StandardGpuResources()
    
    res.setTempMemory(1024*1024*512)
    co = faiss.GpuClonerOptions()
    co.useFloat16 = subvector_num >= 56
    co.verbose = True

    # faiss.omp_set_num_threads(6)
    dim = doc_embed_size
    index = faiss.index_factory(dim, 
        f"OPQ{subvector_num},IVF1,PQ{subvector_num}x8", faiss.METRIC_INNER_PRODUCT)
    index.verbose = True
    index = faiss.index_cpu_to_gpu(res, 0, index, co)    
    logger.info("Starting to train faiss index on document embeddings!")
    assert not index.is_trained
    index.train(doc_embeddings)
    logger.info("Starting to index document embeddings!")
    assert index.is_trained
    index.add(doc_embeddings)
    logger.info("Converting indexes from GPU to CPU!")
    index = faiss.index_gpu_to_cpu(index)
    logger.info("Saving index to CPU memory!")
    faiss.write_index(index, save_index_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocess_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--subvector_num", type=int, required=True)
    parser.add_argument("--model_dir", type=str, default="sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco")
    parser.add_argument("--max_doc_length", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--doc_embed_size", type=int, default=768)
    args = parser.parse_args()

    init(**vars(args))
