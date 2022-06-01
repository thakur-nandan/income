from income.jpq.model import JPQDualEncoder, JPQDualEncoderTASB
from income.jpq.search import DenseRetrievalJPQSearch as DRJS
from income import LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir import util
import random, os, pathlib
import argparse
import logging
import faiss

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--query_encoder", type=str, default="nthakur20/jpq-question_encoder-base-msmarco-distilbert-tas-b")
parser.add_argument("--doc_encoder", type=str, default="nthakur20/jpq-document_encoder-base-msmarco-distilbert-tas-b")
parser.add_argument("--split", type=str, default='test')
parser.add_argument("--encode_batch_size", type=int, default=64)
parser.add_argument("--output_index_path", type=str, default=None)
args = parser.parse_args()

#### Download scifact.zip dataset and unzip the dataset
dataset = args.dataset
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
data_path = util.download_and_unzip(url, out_dir)

#### Provide the data_path where scifact has been downloaded and unzipped
corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=args.split)

#### Load pre-computed index (if present) ####
if args.output_index_path is not None and os.path.isfile(args.output_index_path):
    corpus_index = faiss.read_index(args.output_index_path)
    logging.info("Loaded Corpus Faiss Index: %s\n" % args.output_index_path)
else:
    logging.info("Faiss Index Not Found. Starting to Encode documents")
    corpus_index = None

#### Load the RepCONC model and retrieve using dot-similarity
model = DRJS(JPQDualEncoderTASB((args.query_encoder, args.doc_encoder),), batch_size=args.encode_batch_size, corpus_index=corpus_index)
retriever = EvaluateRetrieval(model, score_function="dot") # or "dot" for dot-product
results = retriever.retrieve(corpus, queries)

#### Saving Faiss Corpus Index into memory ####
if args.output_index_path is not None:
    os.makedirs(os.path.dirname(args.output_index_path), exist_ok=True)
    faiss.write_index(model.corpus_index, args.output_index_path)

#### Evaluate your model with NDCG@k, MAP@K, Recall@K and Precision@K  where k = [1,3,5,10,100,1000] 
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")
recall_cap = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="r_cap")

#### Print top-k documents retrieved ####
top_k = 10

query_id, ranking_scores = random.choice(list(results.items()))
scores_sorted = sorted(ranking_scores.items(), key=lambda item: item[1], reverse=True)
logging.info("Query : %s\n" % queries[query_id])

for rank in range(top_k):
    doc_id = scores_sorted[rank][0]
    # Format: Rank x: ID [Title] Body
    logging.info("Rank %d: %s [%s] - %s\n" % (rank+1, doc_id, corpus[doc_id].get("title"), corpus[doc_id].get("text")))