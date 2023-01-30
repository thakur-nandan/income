cudaNum=1


for dataset in android english gaming gis mathematica physics programmers stats tex unix webmasters wordpress
do
    export MODEL=nthakur/contriever-base-msmarco
    export DATASET_=cqadupstack/${dataset}

    CUDA_VISIBLE_DEVICES=${cudaNum} OMP_NUM_THREADS=6 python evaluate_pca.py \
    --dataset ${DATASET_} \
    --model_path ${MODEL}
    --output_dimension 128
done

for dataset in dbpedia-entity
do
    export MODEL=nthakur/contriever-base-msmarco
    
    CUDA_VISIBLE_DEVICES=${cudaNum} OMP_NUM_THREADS=6 python evaluate_pca.py \
    --dataset ${dataset} \
    --model_path ${MODEL}
    --output_dimension 128
done