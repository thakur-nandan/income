import math
import torch
from torch import nn, Tensor
from typing import Iterable, Dict
from torch.nn import functional as F
import logging
logger = logging.getLogger(__name__)


class MarginDistillationLoss(nn.Module):
    """
    Computes the MSE loss between the computed sentence embedding and a target sentence embedding. This loss
    is used when extending sentence embeddings to new languages as described in our publication
    Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation: https://arxiv.org/abs/2004.09813

    For an example, see the documentation on extending language models to new languages.
    """
    def __init__(self, model, scale: float = 1.0, similarity_fct = 'dot'):
        super(MarginDistillationLoss, self).__init__()
        self.model = model
        self.scale = scale
        assert similarity_fct in ['dot', 'cos_sim']
        self.similarity_fct = similarity_fct
        logger.info(f'Set GPL score function to {similarity_fct}')
        self.loss_fct = nn.MSELoss()

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        # sentence_features: query, positive passage, negative passage
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        embeddings_query = reps[0]
        embeddings_pos = reps[1]
        embeddings_neg = reps[2]

        if self.similarity_fct == 'cosine':
            embeddings_query = F.normalize(embeddings_query, dim=-1)
            embeddings_pos = F.normalize(embeddings_pos, dim=-1)
            embeddings_neg = F.normalize(embeddings_neg, dim=-1)

        scores_pos = (embeddings_query * embeddings_pos).sum(dim=-1) * self.scale
        scores_neg = (embeddings_query * embeddings_neg).sum(dim=-1) * self.scale
        margin_pred = scores_pos - scores_neg

        return self.loss_fct(margin_pred, labels)


class BPRMarginDistillationLoss(nn.Module):
    """
        This loss expects as input a batch consisting of sentence pairs (a_1, p_1), (a_2, p_2)..., (a_n, p_n)
        where we assume that (a_i, p_i) are a positive pair and (a_i, p_j) for i!=j a negative pair.
        For each a_i, it uses all other p_j as negative samples, i.e., for a_i, we have 1 positive example (p_i) and
        n-1 negative examples (p_j). It then minimizes the negative log-likehood for softmax normalized scores.
        This loss function works great to train embeddings for retrieval setups where you have positive pairs (e.g. (query, relevant_doc))
        as it will sample in each batch n-1 negative docs randomly.
        The performance usually increases with increasing batch sizes.
        For more information, see: https://arxiv.org/pdf/1705.00652.pdf
        (Efficient Natural Language Response Suggestion for Smart Reply, Section 4.4)
        You can also provide one or multiple hard negatives per anchor-positive pair by structering the data like this:
        (a_1, p_1, n_1), (a_2, p_2, n_2)
        Here, n_1 is a hard negative for (a_1, p_1). The loss will use for the pair (a_i, p_i) all p_j (j!=i) and all n_j as negatives.
        Example::
            from sentence_transformers import SentenceTransformer,  SentencesDataset, LoggingHandler, losses
            from sentence_transformers.readers import InputExample
            model = SentenceTransformer('distilbert-base-nli-mean-tokens')
            train_examples = [InputExample(texts=['Anchor 1', 'Positive 1']),
                InputExample(texts=['Anchor 2', 'Positive 2'])]
            train_dataset = SentencesDataset(train_examples, model)
            train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
            train_loss = losses.MultipleNegativesRankingLoss(model=model)
    """
    def __init__(self, model, scale: float = 1.0, similarity_fct = 'dot', binary_ranking_loss_margin: float = 2.0, hashnet_gamma: float = 0.1):
        """
        :param model: SentenceTransformer model
        :param scale: Output of similarity function is multiplied by scale value
        :param similarity_fct: similarity function between sentence embeddings. By default, cos_sim. Can also be set to dot product (and then set scale to 1)
        """
        super(BPRMarginDistillationLoss, self).__init__()
        self.global_step = 0
        self.model = model
        self.scale = scale
        assert similarity_fct in ['dot', 'cos_sim']
        self.similarity_fct = similarity_fct
        logger.info(f'Set GPL score function to {similarity_fct}')
        self.hashnet_gamma = hashnet_gamma
        self.margin_loss = nn.MSELoss()
        self.margin_ranking_loss = nn.MarginRankingLoss(margin=binary_ranking_loss_margin)
    
    def convert_to_binary(self, input_repr: Tensor) -> Tensor:
        scale = math.pow((1.0 + self.global_step * self.hashnet_gamma), 0.5)
        return torch.tanh(input_repr * scale)

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        embeddings_a = reps[0]
        embeddings_b = torch.cat([self.convert_to_binary(rep) for rep in reps[1:]])    
        
        # Dense Loss
        scores_pos = (embeddings_a * embeddings_b[0]).sum(dim=-1) * self.scale
        scores_neg = (embeddings_a * embeddings_b[1]).sum(dim=-1) * self.scale
        margin_pred = scores_pos - scores_neg
        dense_loss = self.margin_loss(margin_pred, labels)
        
        # Binary Loss
        
        binary_query_repr = self.convert_to_binary(embeddings_a)
        binary_query_scores = torch.matmul(binary_query_repr, embeddings_b.transpose(0, 1))
        pos_mask = binary_query_scores.new_zeros(binary_query_scores.size(), dtype=torch.bool)
        labels = torch.tensor(range(len(labels)), dtype=torch.long, device=labels.device)

        for n, label in enumerate(labels):
            pos_mask[n, label] = True
        pos_bin_scores = torch.masked_select(binary_query_scores, pos_mask)
        pos_bin_scores = pos_bin_scores.repeat_interleave(embeddings_b.size(0) - 1)
        neg_bin_scores = torch.masked_select(binary_query_scores, torch.logical_not(pos_mask))
        bin_labels = pos_bin_scores.new_ones(pos_bin_scores.size(), dtype=torch.int64)
        binary_loss = self.margin_ranking_loss(
            pos_bin_scores, neg_bin_scores, bin_labels)
        
        self.global_step += 1
        
        return dense_loss + binary_loss


class BPRLoss(nn.Module):
    """
        This loss expects as input a batch consisting of sentence pairs (a_1, p_1), (a_2, p_2)..., (a_n, p_n)
        where we assume that (a_i, p_i) are a positive pair and (a_i, p_j) for i!=j a negative pair.
        For each a_i, it uses all other p_j as negative samples, i.e., for a_i, we have 1 positive example (p_i) and
        n-1 negative examples (p_j). It then minimizes the negative log-likehood for softmax normalized scores.
        This loss function works great to train embeddings for retrieval setups where you have positive pairs (e.g. (query, relevant_doc))
        as it will sample in each batch n-1 negative docs randomly.
        The performance usually increases with increasing batch sizes.
        For more information, see: https://arxiv.org/pdf/1705.00652.pdf
        (Efficient Natural Language Response Suggestion for Smart Reply, Section 4.4)
        You can also provide one or multiple hard negatives per anchor-positive pair by structering the data like this:
        (a_1, p_1, n_1), (a_2, p_2, n_2)
        Here, n_1 is a hard negative for (a_1, p_1). The loss will use for the pair (a_i, p_i) all p_j (j!=i) and all n_j as negatives.
        Example::
            from sentence_transformers import SentenceTransformer,  SentencesDataset, LoggingHandler, losses
            from sentence_transformers.readers import InputExample
            model = SentenceTransformer('distilbert-base-nli-mean-tokens')
            train_examples = [InputExample(texts=['Anchor 1', 'Positive 1']),
                InputExample(texts=['Anchor 2', 'Positive 2'])]
            train_dataset = SentencesDataset(train_examples, model)
            train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
            train_loss = losses.MultipleNegativesRankingLoss(model=model)
    """
    def __init__(self, model, scale: float = 1.0, similarity_fct = "dot", binary_ranking_loss_margin: float = 2.0, hashnet_gamma: float = 0.1):
        """
        :param model: SentenceTransformer model
        :param scale: Output of similarity function is multiplied by scale value
        :param similarity_fct: similarity function between sentence embeddings. By default, cos_sim. Can also be set to dot product (and then set scale to 1)
        """
        super(BPRLoss, self).__init__()
        self.global_step = 0
        self.model = model
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.hashnet_gamma = hashnet_gamma
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.margin_ranking_loss = nn.MarginRankingLoss(margin=binary_ranking_loss_margin)
    
    def convert_to_binary(self, input_repr: Tensor) -> Tensor:
        scale = math.pow((1.0 + self.global_step * self.hashnet_gamma), 0.5)
        return torch.tanh(input_repr * scale)

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        embeddings_a = reps[0]
        embeddings_b = torch.cat([self.convert_to_binary(rep) for rep in reps[1:]])    

        # Dense Loss
        scores = self.similarity_fct(embeddings_a, embeddings_b) * self.scale
        labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)  # Example a[i] should match with b[i]
        dense_loss = self.cross_entropy_loss(scores, labels)
        
        # Binary Loss
        binary_query_repr = self.convert_to_binary(embeddings_a)
        binary_query_scores = torch.matmul(binary_query_repr, embeddings_b.transpose(0, 1))
        pos_mask = binary_query_scores.new_zeros(binary_query_scores.size(), dtype=torch.bool)
        for n, label in enumerate(labels):
            pos_mask[n, label] = True
        pos_bin_scores = torch.masked_select(binary_query_scores, pos_mask)
        pos_bin_scores = pos_bin_scores.repeat_interleave(embeddings_b.size(0) - 1)
        neg_bin_scores = torch.masked_select(binary_query_scores, torch.logical_not(pos_mask))
        bin_labels = pos_bin_scores.new_ones(pos_bin_scores.size(), dtype=torch.int64)
        binary_loss = self.margin_ranking_loss(
            pos_bin_scores, neg_bin_scores, bin_labels)
        
        self.global_step += 1
        
        return dense_loss + binary_loss