import faiss
import torch
import sys, os
import logging
import argparse
from transformers import AutoConfig
from .model import JPQTowerTASB

logger = logging.getLogger(__name__)

def convert_to_robertadot(query_encoder_model, doc_encoder_model, query_faiss_index,
                          doc_faiss_index, M, model_output_dir):
    
    model_info = {"query": {"model": query_encoder_model, "faiss_index": query_faiss_index}, 
                  "doc": {"model": doc_encoder_model, "faiss_index": doc_faiss_index}}
    
    for model_type in model_info:
        model_path = model_info[model_type]["model"]
        faiss_index_path = model_info[model_type]["faiss_index"]
        output_path = os.path.join(model_output_dir, "jpqtower-tasb-{}".format(model_type))

        opq_index = faiss.read_index(faiss_index_path)

        vt = faiss.downcast_VectorTransform(opq_index.chain.at(0))            
        assert isinstance(vt, faiss.LinearTransform)
        opq_transform = faiss.vector_to_array(vt.A).reshape(vt.d_out, vt.d_in)

        ivf_index = faiss.downcast_index(opq_index.index)
        invlists = faiss.extract_index_ivf(ivf_index).invlists
        ls = invlists.list_size(0)
        pq_codes = faiss.rev_swig_ptr(invlists.get_codes(0), ls * invlists.code_size)
        pq_codes = pq_codes.reshape(-1, invlists.code_size)

        centroid_embeds = faiss.vector_to_array(ivf_index.pq.centroids)
        centroid_embeds = centroid_embeds.reshape(ivf_index.pq.M, ivf_index.pq.ksub, ivf_index.pq.dsub)
        coarse_quantizer = faiss.downcast_index(ivf_index.quantizer)
        coarse_embeds = faiss.vector_to_array(coarse_quantizer.xb)
        centroid_embeds += coarse_embeds.reshape(ivf_index.pq.M, -1, ivf_index.pq.dsub)
        coarse_embeds[:] = 0   

        config = AutoConfig.from_pretrained(model_path)
        config.name_or_path = output_path
        config.MCQ_M, config.MCQ_K = ivf_index.pq.M, ivf_index.pq.ksub
        
        model = JPQTowerTASB.from_pretrained(model_path, config=config)

        with torch.no_grad():
            model.centroids.copy_(torch.from_numpy(centroid_embeds))
            model.rotation.copy_(torch.from_numpy(opq_transform))

        model.save_pretrained(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_encoder_model", type=str, required=True)
    parser.add_argument("--doc_encoder_model", type=str, required=True)
    parser.add_argument("--query_faiss_index", type=str, required=True)
    parser.add_argument("--doc_faiss_index", type=str, required=True)
    parser.add_argument("--M", type=int, default=96)
    parser.add_argument("--model_output_dir", type=str, required=True)

    args = parser.parse_args()
    convert_to_robertadot(**vars(args))