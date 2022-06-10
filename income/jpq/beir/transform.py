from beir import util
from tqdm.autonotebook import tqdm
from beir.datasets.data_loader import GenericDataLoader

import pathlib, os, csv
import argparse
import logging
import random

random.seed(42)

#### Just some code to print debug information to stdout
logger = logging.getLogger(__name__)

def preprocessing(text):
    return text.replace("\r", " ").replace("\t", " ").replace("\n", " ").strip()

def transform(
    dataset: str, 
    output_dir: str, 
    prefix: str = None, 
    beir_data_root: str = None, 
    split: str = "train"
):
    if not beir_data_root:
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
        out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
        data_path = util.download_and_unzip(url, out_dir)
    else:
        data_path = os.path.join(beir_data_root, dataset)
    
    if prefix:
        corpus, queries, qrels = GenericDataLoader(data_path, prefix=prefix).load(split=split)
    else:
        corpus, queries, qrels = GenericDataLoader(data_path).load(split=split)
    
    corpus_ids, query_ids = list(corpus), list(queries)
    doc_map, query_map = {}, {}

    for idx, corpus_id in enumerate(corpus_ids): 
        doc_map[corpus_id] = idx

    for idx, query_id in enumerate(query_ids):
        query_map[query_id] = idx

    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("Writing Corpus to file: {}...".format(os.path.join(output_dir, "collection.tsv")))
    with open(os.path.join(output_dir, "collection.tsv"), 'w', encoding="utf-8") as fIn:
        writer = csv.writer(fIn, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        for doc_id in tqdm(corpus_ids, total=len(corpus_ids)):
            doc = corpus[doc_id]
            doc_id_new = doc_map[doc_id]
            writer.writerow([doc_id_new, preprocessing(doc.get("title", "")) + " " + preprocessing(doc.get("text", ""))])

    logger.info("Writing Queries to file: {}...".format(os.path.join(output_dir, "queries.tsv")))
    with open(os.path.join(output_dir, "queries.tsv"), 'w', encoding="utf-8") as fIn:
        writer = csv.writer(fIn, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        for qid, query in tqdm(queries.items(), total=len(queries)):
            qid_new = query_map[qid]
            writer.writerow([qid_new, preprocessing(query)])

    logger.info("Writing Qrels to file: {}...".format(os.path.join(output_dir, "qrels.train.tsv")))
    with open(os.path.join(output_dir, "qrels.train.tsv"), 'w', encoding="utf-8") as fIn:
        writer = csv.writer(fIn, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        for qid, docs in tqdm(qrels.items(), total=len(qrels)):
            for doc_id, score in docs.items():
                qid_new = query_map[qid]
                doc_id_new = doc_map[doc_id]
                writer.writerow([qid_new, 0, doc_id_new, score])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--prefix", type=str, default=None)
    parser.add_argument("--beir_data_root", type=str, default=None)
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()
    transform(**vars(args))