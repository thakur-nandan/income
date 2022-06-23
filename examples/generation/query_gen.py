"""
This code snippet has been borrowed from the BEIR Repository for generating synthetic queries using the ``BeIR/query-gen-msmarco-t5-base-v1`` model.
Warning this code can take quite sometime for generating questions for large datasets. 

If you are looking to reproduce scores, we already publicly provide generated queries for all datasets. Check here: https://huggingface.co/BeIR

This can be used for generating queries for a new dataset or your custom dataset. 
Please look into the BEIR repository for faster multi-gpu solutions.

Usage: python query_gen.py --data_path ./nfcorpus --output_dir ./nfcorpus
Make sure you have the BEIR dataset downloaded and unzipped before running the code.
"""

from ast import parse
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.generation import QueryGenerator as QGen
from beir.generation.models import QGenModel
from beir.retrieval.train import TrainRetriever
from sentence_transformers import SentenceTransformer, losses, models
import os
import argparse


def qgen(data_path, output_dir, generator_name_or_path='BeIR/query-gen-msmarco-t5-base-v1', ques_per_passage=3):
    #### Provide the data_path where nfcorpus has been downloaded and unzipped
    corpus = GenericDataLoader(data_path).load_corpus()

    #### question-generation model loading 
    generator = QGen(model=QGenModel(generator_name_or_path))

    #### Query-Generation using Nucleus Sampling (top_k=25, top_p=0.95) ####
    #### https://huggingface.co/blog/how-to-generate
    #### Prefix is required to seperate out synthetic queries and qrels from original
    prefix = "gen"

    #### Generating 3 questions per passage. 
    #### Reminder the higher value might produce lots of duplicates
    #### Generate queries per passage from docs in corpus and save them in data_path
    generator.generate(corpus, output_dir=output_dir, ques_per_passage=ques_per_passage, prefix=prefix, batch_size=128)
    if not os.path.exists(os.path.join(output_dir, 'corpus.jsonl')):
        os.system(f'cp {data_path}/corpus.jsonl {output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()
    qgen(args.data_path, args.output_dir)