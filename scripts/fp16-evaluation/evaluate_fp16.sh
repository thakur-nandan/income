cudaNum=1

for dataset in scifact nfcorpus arguana scidocs fiqa trec-covid trec-news robust04 quora webis-touche2020 signal1m nq fever climate-fever hotpotqa bioasq
do
    export MODEL=nthakur/contriever-base-msmarco
    
    CUDA_VISIBLE_DEVICES=${cudaNum} OMP_NUM_THREADS=6 python evaluate_fp16.py \
    --dataset ${dataset} \
    --model_path ${MODEL}
done