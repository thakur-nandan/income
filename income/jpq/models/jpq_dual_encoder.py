from .backbones import JPQTowerDistilBert, JPQTowerRoberta, JPQTowerBert
from .util import download_url
from typing import Tuple, List, Dict

import faiss
import os, pathlib
import logging
import numpy as np

logger = logging.getLogger(__name__)

class JPQDualEncoder:
    def __init__(self, model_path: Tuple, backbone: str, sep: str =". ", faiss_path: str = None,
                 dummy_url="https://huggingface.co/nthakur20/dummy-IVFPQ-96M-768D-faiss-index/resolve/main/index.faiss", 
                 **kwargs):
        self.sep = sep
        self.backbone = backbone.lower()
        self.tower = {'distilbert': JPQTowerDistilBert, 'roberta': JPQTowerRoberta, 'bert': JPQTowerBert}
        self.q_model = self.tower[self.backbone].from_pretrained(model_path[0])
        self.doc_model = self.tower[self.backbone].from_pretrained(model_path[1])
        self.faiss_path = faiss_path
        self.dummy_url = dummy_url
    
    def encode_queries(self, queries: List[str], batch_size: int = 16, **kwargs) -> np.ndarray:
        return self.q_model.encode(queries, batch_size=batch_size, **kwargs)
    
    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int = 8, faiss_metric = faiss.METRIC_INNER_PRODUCT, **kwargs) -> faiss.Index:
        # Init fake PQ index
        logger.info("Init dummy PQ index...")
        D, M = self.doc_model.config.hidden_size, self.doc_model.config.MCQ_M

        coarse_quantizer = faiss.IndexFlatL2(D)
        assert self.doc_model.config.MCQ_K == 256
        index = faiss.IndexIVFPQ(coarse_quantizer, D, 1, M, 8, faiss_metric)
        
        if D == 768 and M == 96:
            logger.info("Downloading a IVFPQ dummy index...")
            logger.info("IVFPQ Parameters: Dimension = {} and M = {}".format(D, M))
            
            if not self.faiss_path: 
                faiss_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "faiss")
                os.makedirs(faiss_dir, exist_ok=True)
                self.faiss_path = os.path.join(faiss_dir, "index.faiss")
            
            if not os.path.exists(self.faiss_path):    
                logger.info("Starting to download dummy IVFPQ from url: {}".format(self.dummy_url))
                logging.info("Saving dummy IVFPQ to path: {}".format(self.faiss_path))
                download_url(self.dummy_url, save_path=self.faiss_path)
            
            index = faiss.read_index(self.faiss_path)
            assert index.is_trained == True
        
        else:
            logger.info("Training a IVFPQ dummy index...")
            logger.info("IVFPQ Parameters: Dimension = {} and M = {}".format(D, M))
            # # As we need to provide atleast 9984 for no warnings!
            fake_train_pts = np.random.random((9984, D)).astype(np.float32)
            index.train(fake_train_pts) # fake training

        # ignore coarse quantizer
        coarse_quantizer = faiss.downcast_index(index.quantizer)
        coarse_embeds = faiss.vector_to_array(coarse_quantizer.xb)
        coarse_embeds[:] = 0
        faiss.copy_array_to_vector(coarse_embeds.ravel(), coarse_quantizer.xb)
        # set centroid values 
        logger.info("Set centroid values...")
        doc_centroids = self.doc_model.centroids.data.detach().cpu().numpy()
        faiss.copy_array_to_vector(doc_centroids.ravel(), index.pq.centroids)
        # some other stuffs
        index.precomputed_table.clear()
        index.precompute_table()

        # encode documents and add to index 
        logger.info("Starting to encode documents and add to faiss index...")
        sentences = [(doc["title"] + self.sep + doc["text"]).strip() if "title" in doc else doc["text"].strip() for doc in corpus]
        index = self.doc_model.encode(sentences, batch_size=batch_size, index = index, **kwargs)
        
        # re-set centroid embeddings
        logger.info("Starting to reset centroid embeddings...")
        query_centroids = self.q_model.centroids.data.detach().cpu().numpy()
        faiss.copy_array_to_vector(query_centroids.ravel(), index.pq.centroids)
        return index