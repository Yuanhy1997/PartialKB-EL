split="_t058_sub"
export CUDA_VISIBLE_DEVICES=$1
python -u run_retriever.py \
	--init_model /platform_tech/yuanzheng/GENE2E/entqa/checkpoints/retriever.pt \
	--model /platform_tech/yuanzheng/GENE2E/entqa/run/mm$split/retriever.pt \
	--data_dir /platform_tech/yuanzheng/GENE2E/entqa/medmention/retriever_input$split/ \
    --kb_dir /platform_tech/yuanzheng/GENE2E/entqa/medmention/kb$split/ \
	--k 100 \
	--num_cands 64 \
    --pretrained_path /platform_tech/yuanzheng/GENE2E/entqa/blink/ \
	--gpus $1 \
    --max_len 128   \
	--mention_bsz 4096 \
       	--entity_bsz 2048 \
       	--epochs 30 \
       	--B 4 \
       	--lr 2e-6 \
       	--rands_ratio 0.9  \
	--logging_step 100 \
	--warmup_proportion 0.2 \
       	--out_dir /platform_tech/yuanzheng/GENE2E/entqa/run/mm$split/reader_input/ \
	--gradient_accumulation_steps 2 \
       	--type_loss sum_log_nce \
	--cands_embeds_path /platform_tech/yuanzheng/GENE2E/entqa/run/mm$split/cache_embedding/candidate_embeds.npy \
	--blink \
	--use_title \
       	--add_topic #>> train_mm$split.out &
