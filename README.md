<h1 style="text-align:center">
<img style="vertical-align:middle" width="772" height="180" src="./images/income-logo.png" />
</h1>

## :dollar: What is it?
Index Compression Methods (INCOME) repository helps you easily train and evaluate memory-compressed binary retrievers on any custom dataset. We provide recent state-of-the-art techniques for training and unsupervised (without requiring custom training data) for domain-adaptation of NLP-based binary retrieval models across any dataset. 

We currently support the following models: [BPR: Binary Passage Retriever](https://aclanthology.org/2021.acl-short.123/), [JPQ: ](https://dl.acm.org/doi/10.1145/3459637.3482358) 

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

|   | Link| Models (on HF)| Memory Compr. | BEIR (Avg. NDCG@10) |
|---|:---:|:----:|:----:|:----:|
| **Supervised Compression** |
| BPR | [(Yamada et al., 2021)](https://aclanthology.org/2021.acl-short.123/) | [NQ (DPR)]() | 32x | 0.201 |
| BPR | [(Thakur et al., 2022)](https://arxiv.org/abs/2205.11498) | [TAS-B](https://huggingface.co/nthakur20/bpr-base-msmarco-distilbert-tas-b)  | 32x|  0.357 |
| JPQ | [(Zhan et al., 2021)](https://dl.acm.org/doi/10.1145/3459637.3482358) | STAR [(query)](https://huggingface.co/nthakur20/jpq-question_encoder-base-msmarco-roberta-star) [(doc)](https://huggingface.co/nthakur20/jpq-document_encoder-base-msmarco-roberta-star) | 32x| 0.389 |
| JPQ | [(Thakur et al., 2022)](https://arxiv.org/abs/2205.11498) | TAS-B [(query)](https://huggingface.co/nthakur20/jpq-question_encoder-base-msmarco-distilbert-tas-b) [(doc)](https://huggingface.co/nthakur20/jpq-document_encoder-base-msmarco-distilbert-tas-b)  | 32x | 0.402 |


## :dollar: Quick Example




## :dollar: Why should we do domain adaptation?


## :dollar: Inference


## :dollar: Training


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