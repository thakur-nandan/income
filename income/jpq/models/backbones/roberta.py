import os
import torch
import faiss
import logging
import pathlib
import numpy as np
from torch import nn
from numpy import ndarray
from torch import nn, Tensor
from tqdm.autonotebook import trange
from typing import List, Dict, Union, Tuple

from transformers import RobertaModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
from .roberta_tokenizer import RobertaTokenizer as JPQTokenizer

from .util import pack_tensor_2D, batch_to_device

logger = logging.getLogger(__name__)

class RobertaDot(RobertaPreTrainedModel):
    def __init__(self, config):
        RobertaPreTrainedModel.__init__(self, config)
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.embeddingHead = nn.Linear(config.hidden_size, config.hidden_size)
        self.norm = nn.LayerNorm(config.hidden_size)
        self.apply(self._init_weights)               
    
    def forward(self, input_ids, attention_mask):
        outputs1 = self.roberta(input_ids=input_ids,
                                attention_mask=attention_mask)
        full_emb = outputs1[0][:, 0]
        embeds = self.norm(self.embeddingHead(full_emb))
        return embeds

class JPQTowerRoberta(RobertaDot):
    def __init__(self, config, max_input_length=512):
        super().__init__(config)
        assert config.hidden_size % config.MCQ_M == 0
        self.centroids = nn.Parameter(torch.zeros((
            config.MCQ_M, config.MCQ_K, config.hidden_size // config.MCQ_M)))
        self.rotation = nn.Parameter(torch.eye(config.hidden_size))
        self.tokenizer = RobertaTokenizer.from_pretrained(config._name_or_path, do_lower_case=True)
        self.max_input_length = max_input_length
    
    def forward(self, input_ids, attention_mask):
        unrotate_embeds = super().forward(input_ids, attention_mask)
        rotate_embeds = unrotate_embeds @ self.rotation.T
        return rotate_embeds

    def tokenize(self, texts: List[str]):
        texts = [t[:10000] if len(t) > 0 else " " for t in texts]
        features = self.tokenizer.batch_encode_plus(
            texts, max_length=self.max_input_length)
        features['input_ids'] = pack_tensor_2D(
                features['input_ids'], 
                default=self.tokenizer.pad_token_id, 
                dtype=torch.int64)
        features['attention_mask'] = pack_tensor_2D(
                features['attention_mask'], 
                default=0, 
                dtype=torch.int64)
        return features

    def encode(self, texts: Union[str, List[str]],
               batch_size: int = 32,
               show_progress_bar: bool = None,
               device: str = None, 
               index: faiss.Index = None) -> Union[ndarray, faiss.Index]:
        """
        Computes text embeddings

        :param texts: the texts to embed
        :param batch_size: the batch size used for the computation
        :param show_progress_bar: Output a progress bar when encode texts
        :param device: Which torch.device to use for the computation
        :param index: initial faiss index
        :return:
            Return index if index is given, else return numpy matrix
        """

        self.eval()
        if show_progress_bar is None:
            show_progress_bar = (logger.getEffectiveLevel()==logging.INFO or logger.getEffectiveLevel()==logging.DEBUG)

        input_was_string = False
        if isinstance(texts, str) or not hasattr(texts, '__len__'): #Cast an individual sentence to a list with length 1
            texts = [texts]
            input_was_string = True

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Use pytorch device: {}".format(device))

        self.to(device) # TODO: Utilize mutliple gpus

        all_embeddings = []

        for start_index in trange(0, len(texts), batch_size, desc="Batches", disable=not show_progress_bar):
            sentences_batch = texts[start_index:start_index+batch_size]
            features = self.tokenize(sentences_batch)
            features = batch_to_device(features, device)

            with torch.no_grad():
                embeddings = self.forward(**features)
                embeddings = embeddings.detach().cpu().numpy()

                if index is None:
                    all_embeddings.append(embeddings)
                else:
                    index.add(embeddings)

        if index is None:
            all_embeddings = np.vstack(all_embeddings)
            if input_was_string:
                all_embeddings = all_embeddings[0]
            return all_embeddings
        else:
            return index