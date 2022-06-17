'''
This example shows how to convert a dense retriever into a Binary Passage Retriever (BPR) by finetuning on the MS Marco dataset (https://github.com/microsoft/MSMARCO-Passage-Ranking).
The model is trained using hard negatives which were specially mined with different dense and lexical search methods for MSMARCO. 

The idea for Binary Passage Retriever originated by Yamada et. al, 2021 in Efficient Passage Retrieval with Hashing for Open-domain Question Answering.
For more details, please refer here: https://arxiv.org/abs/2106.00882

This example has been taken from here with few modifications to train SBERT (MSMARCO-v3) models: 
(https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/ms_marco/train_bi-encoder-v3.py)

The queries and passages are passed independently to the transformer network to produce fixed sized binary codes or hashes!!
These embeddings can then be compared using hamming distances to find matching passages for a given query.

For training, we use BPRMarginDistillationLoss (MarginRankingLoss + MultipleNegativesRankingLoss). There, we pass triplets in the format:
(query, positive_passage, negative_passage)

Negative passage are hard negative examples, that were mined using different dense embedding methods and lexical search methods.
Each positive and negative passage comes with a score from a Cross-Encoder. This allows denoising, i.e. removing false negative
passages that are actually relevant for the query.

Running this script:
python convert_dense_to_bpr.py --output_path "./output" --model_name "sentence-transformers/msmarco-distilbert-base-tas-b" --score_function "dot"
'''

from sentence_transformers import SentenceTransformer, InputExample
from beir import util, LoggingHandler
from income.bpr.gpl import BPRMarginDistillationLoss
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.train import TrainRetriever

from torch.utils.data import Dataset
from tqdm.autonotebook import tqdm
from transformers import set_seed
from torch import nn

import pathlib, os, gzip, json
import argparse
import logging
import random
import sys

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

parser = argparse.ArgumentParser()
#### Required parameters
parser.add_argument("--output_path", type=str, required=True, help="Path to where the output BPR model will be stored")
parser.add_argument("--model_name", type=str, required=True, help="Input Sentence Transformer Model")
parser.add_argument("--score_function", type=str, choices=["cosine", "dot"], required=True, help="Similarity Function between document and query: either in ['cosine', 'dot']")

#### Non Required Parameters
parser.add_argument("--train_batch_size", type=int, default=75)
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--max_sequence_length", type=int, default=350)
parser.add_argument("--cross_encoder_margin", type=int, default=3)
parser.add_argument("--num_negatives_per_system", type=int, default=5)
parser.add_argument("--checkpoint_save_steps", type=int, default=5000)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

#### Setting Seed = 42 for reproduction
set_seed(args.seed)

#### Download msmarco.zip dataset and unzip the dataset
dataset = "msmarco"
# url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
# out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
# data_path = util.download_and_unzip(url, out_dir)
data_path = "/home/ukp/thakur/projects/sbert_retriever/datasets-new/{}".format(dataset)

## Load BEIR MSMARCO training dataset, this will be used for query and corpus for reference.
corpus, queries, _ = GenericDataLoader(data_path).load(split="train")

#################################
#### Parameters for Training ####
#################################

train_batch_size = args.train_batch_size            # Increasing the train batch size improves the model performance, but requires more GPU memory (O(n))
max_seq_length = args.max_sequence_length           # Max length for passages. Increasing it, requires more GPU memory (O(n^2))
ce_score_margin = args.cross_encoder_margin         # Margin for the CrossEncoder score between negative and positive passages
num_negs_per_system = args.num_negatives_per_system # We used different systems to mine hard negatives. Number of hard negatives to add from each system

##################################################
#### Download MSMARCO Hard Negs Triplets File ####
##################################################

triplets_url = "https://sbert.net/datasets/msmarco-hard-negatives.jsonl.gz"
msmarco_triplets_filepath = os.path.join(data_path, "msmarco-hard-negatives.jsonl.gz")
if not os.path.isfile(msmarco_triplets_filepath):
    util.download_url(triplets_url, msmarco_triplets_filepath)

#### Load the hard negative MSMARCO jsonl triplets from SBERT 
#### These contain a ce-score which denotes the cross-encoder score for the query and passage.
#### We chose a margin between positive and negative passage scores => above which consider negative as hard negative. 
#### Finally to limit the number of negatives per passage, we define num_negs_per_system across all different systems.

logging.info("Loading MSMARCO hard-negatives...")

train_queries = {}
with gzip.open(msmarco_triplets_filepath, 'rt', encoding='utf8') as fIn:
    for line in tqdm(fIn, total=502939):
        data = json.loads(line)
        
        #Get the positive passage ids
        pos_pids = [item['pid'] for item in data['pos']]
        pos_min_ce_score = min([item['ce-score'] for item in data['pos']])
        ce_score_threshold = pos_min_ce_score - ce_score_margin
        
        #Get the hard negatives
        neg_pids = set()
        for system_negs in data['neg'].values():
            negs_added = 0
            for item in system_negs:
                if item['ce-score'] > ce_score_threshold:
                    continue

                pid = item['pid']
                if pid not in neg_pids:
                    neg_pids.add(pid)
                    negs_added += 1
                    if negs_added >= num_negs_per_system:
                        break
        
        if len(pos_pids) > 0 and len(neg_pids) > 0:
            train_queries[data['qid']] = {'query': queries[data['qid']], 'pos': pos_pids, 'hard_neg': list(neg_pids)}
        
logging.info("Train queries: {}".format(len(train_queries)))

#### We create a custom MSMARCO dataset that returns triplets (query, positive, negative)
#### on-the-fly based on the information from the mined-hard-negatives jsonl file.

class MSMARCODataset(Dataset):
    def __init__(self, queries, corpus):
        self.queries = queries
        self.queries_ids = list(queries.keys())
        self.corpus = corpus

        for qid in self.queries:
            self.queries[qid]['pos'] = list(self.queries[qid]['pos'])
            self.queries[qid]['hard_neg'] = list(self.queries[qid]['hard_neg'])
            random.shuffle(self.queries[qid]['hard_neg'])

    def __getitem__(self, item):
        query = self.queries[self.queries_ids[item]]
        query_text = query['query']

        pos_id = query['pos'].pop(0)    #Pop positive and add at end
        pos_text = self.corpus[pos_id]["text"]
        query['pos'].append(pos_id)

        neg_id = query['hard_neg'].pop(0)    #Pop negative and add at end
        neg_text = self.corpus[neg_id]["text"]
        query['hard_neg'].append(neg_id)

        return InputExample(texts=[query_text, pos_text, neg_text])

    def __len__(self):
        return len(self.queries)



########################################
#### Model Loading for BPR training ####
########################################

model_name = args.model_name
output_path = args.output_path
num_epochs = args.num_epochs
score_function = args.score_function

#### Loading model with SentenceTransformer Model. Currently suppors all "sentence transformer" models.
model = SentenceTransformer(model_name)
model.max_seq_length = max_seq_length

#### Provide a high batch-size to train better with triplets!
retriever = TrainRetriever(model=model, batch_size=train_batch_size)

# For training the SentenceTransformer model, we need a dataset, a dataloader, and a loss used for training.
train_dataset = MSMARCODataset(train_queries, corpus=corpus)
train_dataloader = retriever.prepare_train(train_dataset, shuffle=True, dataset_present=True)

if score_function == "dot":
    train_loss = BPRMarginDistillationLoss(model=retriever.model, similarity_fct=util.dot_score, scale=1)

elif score_function == "cosine":
    train_loss = BPRMarginDistillationLoss(model=retriever.model, similarity_fct=util.cos_sim, scale=20)

#### If no dev set is present from above use dummy evaluator
ir_evaluator = retriever.load_dummy_evaluator()

#### Provide output_path for model to be saved
os.makedirs(output_path, exist_ok=True)

#### Configure Train params
warmup_steps = 1000
checkpoint_path = os.path.join(output_path, "checkpoints")
checkpoint_save_steps = args.checkpoint_save_steps
checkpoint_save_total_limit = 100

### train with Automatic mixed precision (AMP)
retriever.fit(train_objectives=[(train_dataloader, train_loss)],  
                epochs=num_epochs,
                output_path=output_path,
                warmup_steps=warmup_steps,
                checkpoint_path=checkpoint_path,
                checkpoint_save_total_limit=checkpoint_save_total_limit,
                checkpoint_save_steps = checkpoint_save_steps,
                use_amp=True)