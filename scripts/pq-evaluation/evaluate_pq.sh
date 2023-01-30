cudaNum=6

for dataset in fiqa
do
    export MODEL=nthakur/contriever-base-msmarco
    
    CUDA_VISIBLE_DEVICES=${cudaNum} OMP_NUM_THREADS=6 python evaluate_pq.py \
    --dataset ${dataset} \
    --model_path ${MODEL}
done