CudaNum=4

for dataset in msmarco
do
    # Prefix used for storing synthetic query generated for BEIR
    export PREFIX=gen
    export BEIR_DATA_ROOT=/store2/scratch/n3thakur/beir-datasets

    # #### (1) Convert BEIR dataset to JPQ Format ####
    # OMP_NUM_THREADS=6 python -m income.jpq.beir.transform \
    #                   --beir_data_root ${BEIR_DATA_ROOT} \
    #                   --dataset ${dataset} \
    #                   --output_dir "./datasets/${dataset}-train" \

    #### (2) PREPROCESSING script tokenizes the queries and corpus ####
    # CUDA_VISIBLE_DEVICES=${CudaNum} OMP_NUM_THREADS=6 python -m income.jpq.preprocess \
    #                                 --tokenizer "nthakur/contriever-base-msmarco" \
    #                                 --data_dir "./datasets/${dataset}-train" \
    #                                 --out_data_dir "./preprocessed/${dataset}-train"
    
    ### (3) INIT script trains the IVFPQ corpus faiss index ####
    CUDA_VISIBLE_DEVICES=${CudaNum} OMP_NUM_THREADS=6 python -m income.jpq.init \
        --preprocess_dir "./preprocessed/${dataset}-train" \
        --model_dir "nthakur/contriever-base-msmarco" \
        --backbone "bert" \
        --max_doc_length 350 \
        --output_dir "./init/${dataset}-train" \
        --subvector_num 96 \
        --eval_batch_size 256
    
    #### (4) TRAIN script trains TAS-B using Generated Queries (GenQ) ####
    CUDA_VISIBLE_DEVICES=${cudaNum} OMP_NUM_THREADS=6 python -m income.jpq.train_genq \
        --preprocess_dir "./preprocessed/${dataset}-train" \
        --model_save_dir "./final_models/${dataset}-train/genq" \
        --log_dir "./logs/${dataset}-train/log" \
        --init_index_path "./init/${dataset}-train/OPQ96,IVF1,PQ96x8.index" \
        --init_model_path "nthakur/contriever-base-msmarco" \
        --init_backbone "bert" \
        --lambda_cut 200 \
        --centroid_lr 1e-4 \
        --train_batch_size 256 \
        --num_train_epochs 2 \
        --gpu_search \
        --max_seq_length 64

    # #### (5) Convert script converts TASBDot to JPQTower (Required for final evaluation) ####
    # OMP_NUM_THREADS=6 python -m income.jpq.models.jpqtower_converter \
    #         --query_encoder_model "./final_models/${dataset}-train/genq/epoch-2" \
    #         --doc_encoder_model "nthakur/contriever-base-msmarco" \
    #         --query_faiss_index "./final_models/${dataset}-train/genq/epoch-2/OPQ96,IVF1,PQ96x8.index" \
    #         --doc_faiss_index "./init/${dataset}-train/OPQ96,IVF1,PQ96x8.index" \
    #         --model_output_dir "./jpqtower/${dataset}-train/"

done