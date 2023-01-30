cudaNum=4

for dataset in nfcorpus
do
    export QUERY_MODEL=income/jpq-question_encoder-base-msmarco-contriever
    export DOC_MODEL=income/jpq-document_encoder-base-msmarco-contriever
    export BACKBONE=bert

    CUDA_VISIBLE_DEVICES=${cudaNum} OMP_NUM_THREADS=6 python jpq_beir_evaluation.py \
    --dataset ${dataset} \
    --query_encoder ${QUERY_MODEL} \
    --doc_encoder ${DOC_MODEL} \
    --backbone ${BACKBONE}
done