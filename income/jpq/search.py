from typing import Dict
import faiss
import logging

logger = logging.getLogger(__name__)

class DenseRetrievalJPQSearch:
    
    def __init__(self, model, batch_size: int = 128, corpus_index: faiss.Index = None, **kwargs):
        #model is class that provides encode_corpus() and encode_queries()
        self.model = model
        self.batch_size = batch_size
        # Faiss has no cosine similarity metric
        self.score_functions = {'dot': faiss.METRIC_INNER_PRODUCT}
        self.score_function_desc = {'dot': "Dot Product"}
        # Since we use compact Index, this is not required
        # self.corpus_chunk_size = corpus_chunk_size 
        self.show_progress_bar = True #TODO: implement no progress bar if false
        # self.convert_to_tensor = True : Faiss uses numpy

        # so we can reuse stored faiss index
        # and do not have to encode the corpus again
        self.corpus_index = corpus_index
        self.results = {}
    
    def search(self, 
               corpus: Dict[str, Dict[str, str]], 
               queries: Dict[str, str], 
               top_k: int, 
               score_function: str,
                **kwargs) -> Dict[str, Dict[str, float]]:
        #Create embeddings for all queries using model.encode_queries()
        #Runs semantic search against the corpus embeddings
        #Returns a ranked list with the corpus ids
        if score_function not in self.score_functions:
            raise ValueError("score function: {} must be either (dot) for dot product".format(score_function))
            
        logger.info("Encoding Queries...")
        query_ids = list(queries.keys())
        self.results = {qid: {} for qid in query_ids}
        queries = [queries[qid] for qid in queries]
        query_embeddings = self.model.encode_queries(
            queries, 
            batch_size=self.batch_size, 
            show_progress_bar=self.show_progress_bar)
          
        logger.info("Sorting Corpus by document length (Longest first)...")
        corpus_ids = sorted(corpus, key=lambda k: len(corpus[k].get("title", "") + corpus[k].get("text", "")), reverse=True)
        # corpus_ids = list(corpus)
        corpus = [corpus[cid] for cid in corpus_ids]

        if self.corpus_index is None:
            logger.info("Encoding Corpus in batches... Warning: This might take a while!")
            logger.info("Scoring Function: {} ({})".format(self.score_function_desc[score_function], score_function))
            self.corpus_index = self.model.encode_corpus(
                corpus, batch_size=self.batch_size,
                faiss_metric=self.score_functions[score_function],
                show_progress_bar=self.show_progress_bar
            )
        else:
            logger.warning("Skip the corpus encoding process and utilize pre-computed corpus_index")
        
        if faiss.get_num_gpus() == 1:
            logger.info("Transfering index to GPU-0")
            res = faiss.StandardGpuResources()
            co = faiss.GpuClonerOptions()
            co.useFloat16 = faiss.downcast_index(self.corpus_index).pq.M >= 56
            corpus_index = faiss.index_cpu_to_gpu(res, 0, self.corpus_index, co)
        
        elif faiss.get_num_gpus() > 1:
            logger.info("Transfering index to multiple GPUs")
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = faiss.downcast_index(self.corpus_index).pq.M >= 56
            corpus_index = faiss.index_cpu_to_all_gpus(self.corpus_index, co)
        
        # keep self.corpus_index on cpu
        else:
            logger.info("Keeping index in CPU")
            corpus_index = self.corpus_index

        logger.info("Begin search")
        top_k_values, top_k_idx = corpus_index.search(query_embeddings, top_k+1)

        logger.info("Writing results")
        for query_itr in range(len(query_embeddings)):
            query_id = query_ids[query_itr]                  
            for corpus_id_offset, score in zip(top_k_idx[query_itr], top_k_values[query_itr]):
                corpus_id = corpus_ids[corpus_id_offset]
                # Ignore self and empty text
                if corpus_id != query_id and len(corpus[corpus_id_offset]['text'].strip()) > 0:
                    self.results[query_id][corpus_id] = float(score)
        return self.results 