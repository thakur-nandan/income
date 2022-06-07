<h1 style="text-align:center">
<img style="vertical-align:middle" width="772" height="180" src="./images/income-logo.png" />
</h1>

## :dollar: What is it?
Index Compression Methods (INCOME) repository helps you easily train and evaluate different **memory efficient** dense retrievers across any custom dataset. We support all popular compressed dense retriever methods and provide effective techniques for unsupervised domain-adaptation training, open-sourced models and easy evaluation code. 

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

We currently support training and inference of these compressed dense retrievers within our repository:

|   | Models (on HF)| Memory Compr. | BEIR (Avg. NDCG@10) | Index Size | GCP Cloud | Cost per. Month (in \$) |
|:---:|:----:|:----:|:----:|:----:|:----:|:----:|
| **No Compression** |
| TAS-B [(Hofstatter et al., 2021)](https://arxiv.org/abs/2104.06967) | [TAS-B](https://huggingface.co/sentence-transformers/msmarco-distilbert-base-tas-b) | 1 x | 0.413 | 65 GB | n2-highmem-8 | \$306.05 |   
| **Supervised Compression** |
| BPR [(Yamada et al., 2021)](https://aclanthology.org/2021.acl-short.123/) | [NQ (DPR)]() | 32 x | 0.201 | 2.2 GB | n1-standard-1 | \$24.27 |
| BPR [(Thakur et al., 2022)](https://arxiv.org/abs/2205.11498) | [TAS-B](https://huggingface.co/nthakur20/bpr-base-msmarco-distilbert-tas-b)  | 32 x |  **0.357** | 2.2 GB | n1-standard-1 | \$24.27 |
| JPQ [(Zhan et al., 2021)](https://dl.acm.org/doi/10.1145/3459637.3482358) | STAR [(query)](https://huggingface.co/nthakur20/jpq-question_encoder-base-msmarco-roberta-star) [(doc)](https://huggingface.co/nthakur20/jpq-document_encoder-base-msmarco-roberta-star) | 32 x | 0.389 | 2.2 GB | n1-standard-1 | \$24.27 |
| JPQ [(Thakur et al., 2022)](https://arxiv.org/abs/2205.11498) | TAS-B [(query)](https://huggingface.co/nthakur20/jpq-question_encoder-base-msmarco-distilbert-tas-b) [(doc)](https://huggingface.co/nthakur20/jpq-document_encoder-base-msmarco-distilbert-tas-b)  | 32 x | **0.402** | 2.2 GB | n1-standard-1 | \$24.27 |

The Index size and costs are estimated for a user who wants to build a semantic search engine over the English Wikipedia containing about 21 million passages you need to encode. 
Using float16 (and no further compression techniques) and 384 dimensions, the resulting embeddings have a size of about 16GB

## :dollar: Quick Example




## :dollar: Why should we do domain adaptation?



## :dollar: Inference


## :dollar: Training

### :dollar: BPR

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

### :dollar: JPQ





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