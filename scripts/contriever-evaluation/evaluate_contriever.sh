cudaNum=0

for dataset in bioasq
do
    export MODEL=nthakur/contriever-base-msmarco
    
    CUDA_VISIBLE_DEVICES=${cudaNum} OMP_NUM_THREADS=6 python evaluate_contriever.py \
    --dataset ${dataset} \
    --model_path ${MODEL}
done