#--init_model /mnt/data/run/BC5CDR/retriever.pt \
split="_res"
nohup python -u run_retriever.py \
	--init_model /mnt/data/entqa/checkpoints/retriever.pt \
	--model /mnt/data/run/BC5CDR$split/retriever.pt \
	--data_dir /mnt/data/Generative-End2End-IE/dataset/BC5CDR/retriever_input$split/ \
      	--kb_dir /mnt/data/Generative-End2End-IE/dataset/BC5CDR/kb$split/ \
	--k 100 \
	--num_cands 64 \
       	--pretrained_path /mnt/data/blink/ \
	--gpus 4,5,6,7 \
       	--max_len 128   \
	--mention_bsz 4096 \
       	--entity_bsz 2048 \
       	--epochs 50 \
       	--B 4 \
       	--lr 2e-6 \
       	--rands_ratio 0.9  \
	--logging_step 100 \
	--warmup_proportion 0.2 \
       	--out_dir /mnt/data/run/BC5CDR$split/reader_input/ \
	--gradient_accumulation_steps 2 \
       	--type_loss sum_log_nce \
	--cands_embeds_path /mnt/data/run/BC5CDR$split/cache_embedding/candidate_embeds.npy \
	--blink \
	--use_title \
       	--add_topic >> train_bc5cbdr$split.out &
