cudaNum=6

for dataset in scifact nfcorpus
do
    export MODEL=income/bpr-base-msmarco-contriever
    CUDA_VISIBLE_DEVICES=${cudaNum} OMP_NUM_THREADS=6 python bpr_beir_evaluation.py \
    --dataset ${dataset} \
    --model_path ${MODEL}
done