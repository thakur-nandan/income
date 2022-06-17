"""
This code shows how to download already available generated queries for BEIR datasets
This code requires datasets library to be already installed: ``pip install datasets``

This code will download and format the generated queries in the BEIR format used for GPL training.

Simple Usage: python query_gen_hf --dataset nfcorpus
"""

import datasets
import logging
import os
import pathlib
from tqdm.autonotebook import tqdm
from beir import util

logger = logging.getLogger(__name__)

def get_generated_queries(dataset: str, out_dir: str = None, prefix: str = "gen"):
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
    data_path = util.download_and_unzip(url, out_dir)
    
    ### Loading generated queries from HF
    ### Checkout complete list: https://huggingface.co/BeIR
    logger.info("Downloading generated queries from HF path: BeIR/{}-generated-queries".format(dataset))
    gen_queries_hf = datasets.load_dataset("BeIR/{}-generated-queries".format(dataset))["train"]

    gen_queries, gen_qrels = {}, {}

    for idx, metadata in enumerate(tqdm(gen_queries_hf, total=len(gen_queries_hf))):
        gen_query_id = prefix + "q" + str(idx)
        gen_queries[gen_query_id] = metadata["query"]
        gen_qrels[gen_query_id] = {metadata["_id"]: 1}
    
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
    parser.add_argument('--dataset', format=str, required=True)
    parser.add_argument('--output_dir', format=str, default=None)
    parser.add_argument('--prefix', format=str, default="gen")
    args = parser.parse_args()
    get_generated_queries(args.dataset, args.output_dir, args.prefix)