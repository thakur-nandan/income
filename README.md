<h1 style="text-align:center">
<img style="vertical-align:middle" width="772" height="180" src="./images/income-logo.png" />
</h1>

## :dollar: What is it?
Index Compression Methods (INCOME) repository helps you easily train and evaluate different **memory efficient** dense retrievers across any custom dataset. The pre-trained models produce float embeddings of sizes from between 512 - 1024. However, when storing a large number of embeddings within an index for fast inference, this requires quite a lot of memory / storage. 


In this repository, we focus on index compression and provide models which produce binary embeddings i.e. 1 or -1 which require less dimensions and help you **save both storage and money** on hosting such models in a practical setup with limited money.


We currently support the following memory efficient dense retriever model architectures: 
- [BPR: Binary Passage Retriever](https://aclanthology.org/2021.acl-short.123/) (ACL 2021)
- [JPQ: Jointly Optimizing Query Encoder and Product Quantization to Improve Retrieval Performance](https://dl.acm.org/doi/10.1145/3459637.3482358) (CIKM 2021)

For more information, checkout our publication:
- [Domain Adaptation for Memory-Efficient Dense Retrieval](https://arxiv.org/abs/2205.11498/) (Arxiv preprint)

## :dollar: Installation
One can either install income via `pip`
```bash
pip install income
```
or via source using `git clone`
```bash
$ git clone https://github.com/Nthakur20/income.git
$ cd income
$ pip install -e .
```
With that, you should be ready to go!

## :dollar: Models Supported

We currently support training and inference of these compressed dense retrievers within our repository. We compare the performance and cost of hosting these models below:

|   | Backbone| MSMARCO | BEIR | Memory Size | Query Time | GCP Cloud | Cost per. Month (in \$) |
|:---:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| **No Compression** |
| TAS-B [(Hofstatter et al., 2021)](https://arxiv.org/abs/2104.06967) | [TAS-B](https://huggingface.co/sentence-transformers/msmarco-distilbert-base-tas-b) | 0.408 | 0.415 | 65 GB (1x) | 456.9 ms | n2-highmem-8 | \$306.05 |   
| TAS-B + HNSW [(Hofstatter et al., 2021)](https://arxiv.org/abs/2104.06967) | [TAS-B](https://huggingface.co/sentence-transformers/msmarco-distilbert-base-tas-b)| 0.408 | 0.415 | 151 GB (1x) | 1.8 ms | n2-highmem-32 | \$1224.19 |
| TAS-B + PQ [(Hofstatter et al., 2021)](https://arxiv.org/abs/2104.06967) | [TAS-B](https://huggingface.co/sentence-transformers/msmarco-distilbert-base-tas-b) | 0.358 | 0.361 | 2 GB (32x) | 44.0 ms | n1-standard-1 | \$24.27 |   
| **Supervised Compression: BPR** |
| BPR (TAS-B) [(Thakur et al., 2022)](https://arxiv.org/abs/2205.11498) | [TAS-B](https://huggingface.co/income/bpr-base-msmarco-distilbert-tas-b)  | 0.397 |  0.357 | 2.2 GB (32x) | 38.1 ms |n1-standard-1 | \$24.27 |
| BPR+GenQ (TAS-B) [(Thakur et al., 2022)](https://arxiv.org/abs/2205.11498) | [TAS-B](https://huggingface.co/income/)  | 0.397 | 0.377  | 2.2 GB (32x) | 38.1 ms |n1-standard-1 | \$24.27 |
| BPR+GPL (TAS-B) [(Thakur et al., 2022)](https://arxiv.org/abs/2205.11498) | [TAS-B](https://huggingface.co/income/)  | 0.397 |  **0.398** | 2.2 GB (32x) | 38.1 ms |n1-standard-1 | \$24.27 |
| **Supervised Compression: JPQ** |
| JPQ (TAS-B) [(Thakur et al., 2022)](https://arxiv.org/abs/2205.11498) | TAS-B [(query)](https://huggingface.co/income/jpq-question_encoder-base-msmarco-distilbert-tas-b) [(doc)](https://huggingface.co/income/jpq-document_encoder-base-msmarco-distilbert-tas-b) | 0.400  | 0.402 | 2.2 GB (32x) | 44.0 ms | n1-standard-1 | \$24.27 |
| JPQ+GenQ (TAS-B) [(Thakur et al., 2022)](https://arxiv.org/abs/2205.11498) | TAS-B [(query)](https://huggingface.co/income/jpq-question_encoder-base-msmarco-distilbert-tas-b) [(doc)](https://huggingface.co/income/jpq-document_encoder-base-msmarco-distilbert-tas-b) | 0.400  | 0.417 | 2.2 GB (32x) | 44.0 ms | n1-standard-1 | \$24.27 |
| JPQ+GPL (TAS-B) [(Thakur et al., 2022)](https://arxiv.org/abs/2205.11498) | TAS-B [(query)](https://huggingface.co/income/jpq-question_encoder-base-msmarco-distilbert-tas-b) [(doc)](https://huggingface.co/income/jpq-document_encoder-base-msmarco-distilbert-tas-b) | 0.400  | **0.435** | 2.2 GB (32x) | 44.0 ms | n1-standard-1 | \$24.27 |

The scores denote the NDCG@10 performance of the model. The Index size and costs are estimated for a user who wants to build a semantic search engine over the English Wikipedia containing about 21 million passages you need to encode. 
Using float32 (and no further compression techniques) and 768 dimensions, the resulting embeddings have a size of about 65GB. The ``n2-highmem-8`` server can provide upto 64 GB of memory, whereas the ``n1-standard-1`` server can provide upto 3.75 GB of memory. 

## :dollar: Easily compress your dense retriever

Our technique can easily wrap around any HF-based dense retriever and convert them into a BPR or JPQ based model. Overall, we find the stronger the backbone dense retriever in generalization, the better the BPR and JPQ models. We recently converted these new models and made them available publicly on HF. Incase, you wish to convert your model open a pull request or follow the scripts below for BPR and JPQ seperately.

|   | Backbone| MSMARCO | BEIR | Memory Size | Query Time | GCP Cloud | Cost per. Month (in \$) |
|:---:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|  
| **Supervised Compression** |
| BPR (Contriever) [(Izacard et al., 2021)](https://arxiv.org/abs/2112.09118) | [Contriever](https://huggingface.co/income/bpr-base-msmarco-contriever) | 0.407 | 0.367 | 2.2 GB (32x) | 38.1 ms | n1-standard-1 | \$24.27 |
| BPR (DPR) [(Yamada et al., 2021)](https://aclanthology.org/2021.acl-short.123/) | [NQ (DPR)]() | 0.130 | 0.201 | 2.2 GB (32x) | 38.1 ms | n1-standard-1 | \$24.27 |
| JPQ (STAR) [(Zhan et al., 2021)](https://dl.acm.org/doi/10.1145/3459637.3482358) | STAR [(query)](https://huggingface.co/income/jpq-question_encoder-base-msmarco-roberta-star) [(doc)](https://huggingface.co/income/jpq-document_encoder-base-msmarco-roberta-star) | 0.402 | 0.389 | 2.2 GB (32x) | 44.0 ms | n1-standard-1 | \$24.27 |


## :dollar: Using the INCOME library
The ``income`` library can be used to learn various different vector compression strategies for information retrieval. These can be used 

## :dollar: BPR Model (Getting Started)
This section would introduce few quick examples to train and evaluate BPR models on any custom data you wish to search on.

### Training using GPL: Generative Pseudo Labeling

```bash
export dataset="nfcorpus"

python -m income.bpr.train \
    --path_to_generated_data "generated/$dataset" \
    --base_ckpt "msmarco-distilbert-base-tas-b" \
    --gpl_score_function "dot" \
    --batch_size_gpl 32 \
    --gpl_steps 10000 \
    --new_size -1 \
    --queries_per_passage -1 \
    --output_dir "output/$dataset" \
    --generator "BeIR/query-gen-msmarco-t5-base-v1" \
    --retrievers "msmarco-distilbert-base-tas-b" "msmarco-distilbert-base-v3" "msmarco-MiniLM-L-6-v3" \
    --retriever_score_functions "dot" "cos_sim" "cos_sim" \
    --cross_encoder "cross-encoder/ms-marco-MiniLM-L-6-v2" \
    --qgen_prefix "gen-t5-base-2-epoch-default-lr-3-ques" \
    --evaluation_data "./$dataset" \
    --evaluation_output "evaluation/$dataset" \
    --do_evaluation \
    --use_amp   # Use this for efficient training if the machine supports AMP
```

### Inference

```python
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import BinaryFaissSearch
from beir import util
from income.bpr.model import BPR

dataset = "nfcorpus"
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
data_path = util.download_and_unzip(url, out_dir)

corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
faiss_search = BinaryFaissSearch(BPR("income/bpr-base-msmarco-distilbert-tas-b"), batch_size=128)

retriever = EvaluateRetrieval(faiss_search, score_function="dot")
results = retriever.retrieve(corpus, queries, rerank=True, binary_k=1000)
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
```

## :dollar: JPQ Model (Getting Started)

This section would introduce few quick examples to train and evaluate JPQ models on any custom data you wish to search on.

### Training using GPL: Generative Pseudo Labeling

Training using JPQ and GPL occurs in four steps:

1. Preprocess Dataset to JPQ-friendly format
```bash
export dataset="nfcorpus"
export PREFIX="gen"

python -m income.jpq.beir.transform \
          --dataset ${dataset} \
          --output_dir "./datasets/${dataset}" \
          --prefix  ${PREFIX} \
```
2. Preprocessing Script tokenizes the queries and corpus
```bash
CUDA_VISIBLE_DEVICES=0 python -m income.jpq.preprocess \
                                --data_dir "./datasets/${dataset}" \
                                --out_data_dir "./preprocessed/${dataset}"
    
```
3. INIT script trains the IVFPQ corpus faiss index
```bash
CUDA_VISIBLE_DEVICES=0 python -m income.jpq.init \
  --preprocess_dir "./preprocessed/${dataset}" \
  --model_dir "sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco"
  --max_doc_length 350 \
  --output_dir "./init/${dataset}" \
  --subvector_num 96
```
4. TRAIN script trains TAS-B using Generative Pseudo Labeling (GPL)
```bash
CUDA_VISIBLE_DEVICES=0 python -m income.jpq.train_gpl \
    --preprocess_dir "./preprocessed/${dataset}" \
    --model_save_dir "./final_models/${dataset}/gpl" \
    --log_dir "./logs/${dataset}/log" \
    --init_index_path "./init/${dataset}/OPQ96,IVF1,PQ96x8.index" \
    --init_model_path "sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco" \
    --data_path "./datasets/${dataset}" \
    --cross_encoder "cross-encoder/ms-marco-MiniLM-L-6-v2" \
    --lambda_cut 200 \
    --centroid_lr 1e-4 \
    --train_batch_size 32 \
    --num_train_epochs 2 \
    --gpu_search \
    --max_seq_length 64 \
    --loss_neg_topK 25
```
5. Convert TAS-B trained model into JPQTower (Reqd. for inference)
```bash
python -m income.jpq.models.jpqtower_converter \
        --query_encoder_model "./final_models/${dataset}/genq/epoch-1" \
        --doc_encoder_model "sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco" \
        --query_faiss_index "./final_models/${dataset}/genq/epoch-1/OPQ96,IVF1,PQ96x8.index" \
        --doc_faiss_index "./init/${dataset}/OPQ96,IVF1,PQ96x8.index" \
        --model_output_dir "./jpqtower/${dataset}/"
```
### Inference
```python
from income.jpq.models import JPQDualEncoder
from income.jpq.search import DenseRetrievalJPQSearch as DRJS
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir import util

dataset = "nfcorpus"
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
data_path = util.download_and_unzip(url, out_dir)

corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
model = DRJS(JPQDualEncoder((
    "income/jpq-question_encoder-base-msmarco-distilbert-tas-b", 
    "income/jpq-document_encoder-base-msmarco-distilbert-tas-b"
  ), backbone="distilbert"))

retriever = EvaluateRetrieval(model, score_function="dot") 
results = retriever.retrieve(corpus, queries)
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
```

## :dollar: Reproduction Scripts with TAS-B

|   | Script | BEIR (Avg. NDCG@10) | Memory Size |
|:---:|:----:|:----:|:----:|
| **Baselines** |
| fp-16 | [evaluate_fp16.py](https://github.com/NThakur20/income/blob/development/examples/baselines/evaluate_fp16.py) | 0.414  | 33 GB (2x)   |
| fp-8 | [evaluate_fp16.py](https://github.com/NThakur20/income/blob/development/examples/baselines/evaluate_fp16.py)  | 0.407  | 16 GB (4x)   |
| PCA  | [evaluate_pca.py](https://github.com/NThakur20/income/blob/development/examples/baselines/evaluate_pca.py)    | 0.235  | 22 GB (3x)   |
| TLDR | [evaluate_pca.py](https://github.com/NThakur20/income/blob/development/examples/baselines/evaluate_pca.py)    | 0.240  | 22 GB (3x)   |
| PQ   | [evaluate_pq.py](https://github.com/NThakur20/income/blob/development/examples/baselines/evaluate_pq.py)      | 0.361  | 2.2 GB (32x) |    
| **Supervised Compression**|
| BPR   | [bpr_beir_evaluation.py](https://github.com/NThakur20/income/blob/development/examples/bpr/evaluation/bpr_beir_evaluation.py) | 0.357  | 2.2 GB (32x) |
| JPQ   | [jpq_beir_evaluation.py](https://github.com/NThakur20/income/blob/development/examples/jpq/evaluation/jpq_beir_evaluation.py) | 0.402  | 2.2 GB (32x) |


## :dollar: Why should we do domain adaptation?

![](images/domain-adaptation.png)



### :dollar: Disclaimer

For reproducibility purposes, we work with the original code repositories and modify them in INCOME if they available, for eg. [BPR](https://github.com/studio-ousia/bpr) and [JPQ](https://github.com/jingtaozhan/JPQ). It remains the user's responsibility to determine whether you as a user have permission to use the original models and to cite the right owner of each model. Check the below table for reference.

If you're a model owner and wish to update any part of it, or do not want your model to be included in this library, feel free to post an issue here or make a pull request!

| Model/Method | Citation | GitHub |
|:---:|:----:|:----:|
| BPR | [(Yamada et al., 2021)](https://aclanthology.org/2021.acl-short.123/) | [https://github.com/studio-ousia/bpr](https://github.com/studio-ousia/bpr) |
| JPQ | [(Zhan et al., 2021)](https://dl.acm.org/doi/10.1145/3459637.3482358) | [https://github.com/jingtaozhan/JPQ](https://github.com/jingtaozhan/JPQ) |
| GPL | [(Wang et al., 2021)](https://arxiv.org/abs/2112.07577) | [https://github.com/UKPLab/gpl](https://github.com/UKPLab/gpl)|


## :dollar: Citing & Authors
If you find this repository helpful, feel free to cite our recent publication: [Domain Adaptation for Memory-Efficient Dense Retrieval](https://arxiv.org/abs/2205.11498/):

```
@article{thakur2022domain,
  title={Domain Adaptation for Memory-Efficient Dense Retrieval},
  author={Thakur, Nandan and Reimers, Nils and Lin, Jimmy},
  journal={arXiv preprint arXiv:2205.11498},
  year={2022},
  url={https://arxiv.org/abs/2205.11498/}
}
```

The main contributors of this repository are:
- [Nandan Thakur](https://github.com/Nthakur20), Personal Website: [nandan-thakur.com](https://nandan-thakur.com)

Contact person: Nandan Thakur, [nandant@gmail.com](mailto:nandant@gmail.com)

Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions.

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.