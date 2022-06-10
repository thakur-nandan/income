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