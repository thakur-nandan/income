cudaNum=0

for dataset in scidocs signal1m nq hotpotqa dbpedia-entity
do
    export MODEL=nthakur/contriever-base-msmarco
    
    CUDA_VISIBLE_DEVICES=${cudaNum} OMP_NUM_THREADS=6 python evaluate_fp8.py \
    --dataset ${dataset} \
    --model_path ${MODEL}
done