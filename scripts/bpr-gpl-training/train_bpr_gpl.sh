cudaNum=3

for dataset in scifact scidocs fiqa nfcorpus arguana signal1m nq fever climate-fever
do
    python query_gen_hf.py --dataset ${dataset}

    CUDA_VISIBLE_DEVICES=${cudaNum} python -m income.bpr.train \
        --path_to_generated_data "/store2/scratch/n3thakur/beir-datasets/${dataset}" \
        --base_ckpt "nthakur/contriever-base-msmarco" \
        --base_score_function "dot" \
        --batch_size_gpl 32 \
        --gpl_steps 140000 \
        --new_size -1 \
        --queries_per_passage 3 \
        --output_dir "./output/${dataset}" \
        --generator "BeIR/query-gen-msmarco-t5-base-v1" \
        --retrievers "msmarco-distilbert-base-tas-b" "msmarco-distilbert-base-v3" "nthakur/contriever-base-msmarco" \
        --retriever_score_functions "dot" "cos_sim" "dot" \
        --cross_encoder "cross-encoder/ms-marco-MiniLM-L-6-v2" \
        --qgen_prefix "gen-top-3" \
        --evaluation_data "./${dataset}" \
        --evaluation_output "evaluation/${dataset}" \
        --do_evaluation \
        --use_amp   # Use this for efficient training if the machine supports AMP
done


# for dataset in trec-covid trec-news robust04 quora webis-touche2020 signal1m nq fever climate-fever hotpotqa dbpedia-entity bioasq
# do
#     python query_gen_hf.py --dataset ${dataset}

#     python -m income.bpr.train \
#         --path_to_generated_data "/store2/scratch/n3thakur/beir-datasets/${dataset}" \
#         --base_ckpt "nthakur/contriever-base-msmarco" \
#         --base_score_function "dot" \
#         --batch_size_gpl 32 \
#         --gpl_steps 50000 \
#         --new_size -1 \
#         --queries_per_passage 3 \
#         --output_dir "./output/${dataset}" \
#         --generator "BeIR/query-gen-msmarco-t5-base-v1" \
#         --retrievers "msmarco-distilbert-base-tas-b" "msmarco-distilbert-base-v3" "nthakur/contriever-base-msmarco" \
#         --retriever_score_functions "dot" "cos_sim" "dot" \
#         --cross_encoder "cross-encoder/ms-marco-MiniLM-L-6-v2" \
#         --qgen_prefix "gen-top-3" \
#         --evaluation_data "./${dataset}" \
#         --evaluation_output "evaluation/${dataset}" \
#         --do_evaluation \
#         --use_amp   # Use this for efficient training if the machine supports AMP
# done