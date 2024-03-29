from income.jpq.models import JPQDualEncoder
from income.jpq.search import DenseRetrievalJPQSearch as DRJS
from income import LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir import util
import random, os, pathlib
import argparse
import logging
import faiss
import json

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True)

# ### TAS-B (DistilBERT) backbone based JPQ question and document encoders
# parser.add_argument("--query_encoder", type=str, default="income/jpq-question_encoder-base-msmarco-distilbert-tas-b")
# parser.add_argument("--doc_encoder", type=str, default="income/jpq-document_encoder-base-msmarco-distilbert-tas-b")
# parser.add_argument("--backbone", type=str, default='distilbert', choices=['distilbert', 'roberta', 'bert'])

#### STAR (Roberta) backbone based JPQ question and document encoders
# parser.add_argument("--query_encoder", type=str, default="income/jpq-question_encoder-base-msmarco-roberta-star")
# parser.add_argument("--doc_encoder", type=str, default="income/jpq-document_encoder-base-msmarco-roberta-star")
# parser.add_argument("--backbone", type=str, default='roberta', choices=['distilbert', 'roberta', 'bert'])

### Contriever (BERT) backbone based JPQ question and document encoders
parser.add_argument("--query_encoder", type=str, default="income/jpq-question_encoder-base-msmarco-contriever")
parser.add_argument("--doc_encoder", type=str, default="income/jpq-document_encoder-base-msmarco-contriever")
parser.add_argument("--backbone", type=str, default='bert', choices=['distilbert', 'roberta', 'bert'])

parser.add_argument("--split", type=str, default='test')
parser.add_argument("--encode_batch_size", type=int, default=128)
parser.add_argument("--output_index_path", type=str, default=None)
parser.add_argument("--prefix", type=str, default=None)
args = parser.parse_args()

#### Download scifact.zip dataset and unzip the dataset
dataset = args.dataset
# url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
# out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
# data_path = util.download_and_unzip(url, out_dir)
data_path = os.path.join("/store2/scratch/n3thakur/beir-datasets", dataset)

#### Provide the data_path where scifact has been downloaded and unzipped
if dataset != "msmarco":
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
else:
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="dev")

#### Load pre-computed index (if present) ####
if args.output_index_path is not None and os.path.isfile(args.output_index_path):
    corpus_index = faiss.read_index(args.output_index_path)
    logging.info("Loaded Corpus Faiss Index: %s\n" % args.output_index_path)
else:
    logging.info("Faiss Index Not Found. Starting to Encode documents")
    corpus_index = None

#### Load the RepCONC model and retrieve using dot-similarity
logging.info("JPQ Question Encoder with {} backbone: {}".format(args.backbone, args.query_encoder))
logging.info("JPQ Document Encoder with {} backbone: {}".format(args.backbone, args.doc_encoder))

model = DRJS(JPQDualEncoder((args.query_encoder, args.doc_encoder), backbone=args.backbone), batch_size=args.encode_batch_size, corpus_index=corpus_index)
retriever = EvaluateRetrieval(model, score_function="dot") # or "dot" for dot-product
results = retriever.retrieve(corpus, queries)

# #### Saving Faiss Corpus Index into memory ####
# if args.output_index_path is not None:
#     os.makedirs(os.path.dirname(args.output_index_path), exist_ok=True)
#     faiss.write_index(model.corpus_index, args.output_index_path)

output_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), dataset, "with_gpu")

#### Evaluate your model with NDCG@k, MAP@K, Recall@K and Precision@K  where k = [1,3,5,10,100,1000] 
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")
recall_cap = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="r_cap")
hole = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="hole")

os.makedirs(output_dir, exist_ok=True)
model_path = args.doc_encoder.replace('/', '-')

with open(os.path.join(output_dir, f'{model_path}-metrics.json'), 'w') as f:
    metrics = {
        'nDCG': ndcg,
        'MAP': _map,
        'Recall': recall_cap,
        'Precision': precision,
        'mrr': mrr,
        'hole': hole
    }
    json.dump(metrics, f, indent=4)

# #### Print top-k documents retrieved ####
# top_k = 10

# query_id, ranking_scores = random.choice(list(results.items()))
# scores_sorted = sorted(ranking_scores.items(), key=lambda item: item[1], reverse=True)
# logging.info("Query : %s\n" % queries[query_id])

# for rank in range(top_k):
#     doc_id = scores_sorted[rank][0]
#     # Format: Rank x: ID [Title] Body
#     logging.info("Rank %d: %s [%s] - %s\n" % (rank+1, doc_id, corpus[doc_id].get("title"), corpus[doc_id].get("text")))