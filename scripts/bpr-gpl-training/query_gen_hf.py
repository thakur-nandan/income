"""
This code shows how to download already available generated queries for BEIR datasets
This code requires datasets library to be already installed: ``pip install datasets``

This code will download and format the generated queries in the BEIR format used for GPL training.

Simple Usage: python query_gen_hf --dataset nfcorpus
"""

import datasets
import logging
import argparse
import os
import pathlib
from tqdm.autonotebook import tqdm
from beir import util
import argparse

logger = logging.getLogger(__name__)

def get_generated_queries(dataset: str, out_dir: str = None, prefix: str = "gen"):
    # url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    # out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
    # data_path = util.download_and_unzip(url, out_dir)
    data_path = f"/store2/scratch/n3thakur/beir-datasets/{dataset}"
    
    ### Loading generated queries from HF
    ### Checkout complete list: https://huggingface.co/BeIR
    logger.info("Downloading generated queries from HF path: BeIR/{}-generated-queries".format(dataset))
    gen_queries_hf = datasets.load_dataset("income/{}-top-20-gen-queries".format(dataset))["train"]

    gen_queries, gen_qrels = {}, {}

    for metadata in tqdm(gen_queries_hf, total=len(gen_queries_hf)):
        doc_id = metadata['id']
        for idx, query in enumerate(metadata["queries"]):
            if idx <= 2:
                qid = str(doc_id) + "-genQ"+ str(idx)
                gen_queries[qid] = query
                gen_qrels[qid] = {doc_id: 1}
        
    
    ### Make the qrels folder
    os.makedirs(os.path.join(data_path, prefix + "-qrels"), exist_ok=True)
    qrels_filepath = os.path.join(data_path, prefix + "-qrels", "train.tsv")

    ### Write qrels to generated qrels folder as train.tsv
    logging.info("Storing generated qrels to path: {}".format(qrels_filepath))
    util.write_to_tsv(output_file=qrels_filepath, data=gen_qrels)

    ### Write queries into generated queries file
    queries_filepath = os.path.join(data_path, prefix + "-queries.jsonl")
    logging.info("Storing generated queries to path: {}".format(queries_filepath))
    util.write_to_json(output_file=queries_filepath, data=gen_queries)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--prefix', type=str, default="gen-top-3")
    args = parser.parse_args()
    get_generated_queries(args.dataset, args.output_dir, args.prefix)