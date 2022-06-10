import torch
import faiss
import logging
import numpy as np
from torch import nn
from numpy import ndarray
from torch import nn, Tensor
from tqdm.autonotebook import trange
from typing import List, Dict, Union, Tuple

from transformers import BertModel
from transformers.models.bert.modeling_bert import BertPreTrainedModel

from .util import pack_tensor_2D, batch_to_device
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class BertDot(BertPreTrainedModel):
    def __init__(self, config, pooling="mean"):
        BertPreTrainedModel.__init__(self, config)
        self.bert = BertModel(config, add_pooling_layer=False)
        if not hasattr(config, "pooling"):
            self.config.pooling = pooling
    
    def forward(self, input_ids, attention_mask, normalize=False):
        outputs = self.bert(input_ids, attention_mask, return_dict=True)
        
        if self.config.pooling == "cls":
            emb = outputs.last_hidden_state[:, 0]
        elif self.config.pooling == "mean":
            emb = mean_pooling(outputs, attention_mask)

        if normalize:
            emb = torch.nn.functional.normalize(emb, dim=-1)
        
        return emb

class JPQTowerBert(BertDot):
    def __init__(self, config, max_input_length=512):
        super().__init__(config)
        assert config.hidden_size % config.MCQ_M == 0
        self.centroids = nn.Parameter(torch.zeros((
            config.MCQ_M, config.MCQ_K, config.hidden_size // config.MCQ_M)))
        self.rotation = nn.Parameter(torch.eye(config.hidden_size))
        self.tokenizer = AutoTokenizer.from_pretrained(config._name_or_path)
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
            del features['token_type_ids']
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