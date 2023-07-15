split="_t038_sub"
export CUDA_VISIBLE_DEVICES=$1
python run_reader.py  \
	--init_model /platform_tech/yuanzheng/GENE2E/entqa/checkpoints/reader.pt \
	--model /platform_tech/yuanzheng/GENE2E/entqa/run/mm$split/reader.pt \
	--data_dir /platform_tech/yuanzheng/GENE2E/entqa/run/mm$split/reader_input/  \
       	--kb_dir /platform_tech/yuanzheng/GENE2E/entqa/medmention/kb$split/ \
	--C 64 \
       	--B 2 \
       	--L 180 \
       	--C_val 100 \
       	--gpus $1 \
      	--val_bsz 32 \
	--gradient_accumulation_steps 2 \
       	--warmup_proportion 0.06  \
	--epochs 30 \
       	--lr 5e-6 \
	--thresd 0.05 \
       	--logging_steps 50 \
	--k 3 \
       	--stride 16 \
	--max_passage_len 128 \
       	--filter_span  \
	--type_encoder squad2_electra_large  \
	--type_span_loss sum_log \
       	--type_rank_loss sum_log  \
	--do_rerank \
       	--add_topic \
	--use_title \
	--thresd 0.01 \
	--fp16 \
       	--results_dir /platform_tech/yuanzheng/GENE2E/entqa/run/mm$split/reader_results/


split="_t058_sub"
export CUDA_VISIBLE_DEVICES=$1
python run_reader.py  \
	--init_model /platform_tech/yuanzheng/GENE2E/entqa/checkpoints/reader.pt \
	--model /platform_tech/yuanzheng/GENE2E/entqa/run/mm$split/reader.pt \
	--data_dir /platform_tech/yuanzheng/GENE2E/entqa/run/mm$split/reader_input/  \
       	--kb_dir /platform_tech/yuanzheng/GENE2E/entqa/medmention/kb$split/ \
	--C 64 \
       	--B 2 \
       	--L 180 \
       	--C_val 100 \
       	--gpus $1 \
      	--val_bsz 32 \
	--gradient_accumulation_steps 2 \
       	--warmup_proportion 0.06  \
	--epochs 30 \
       	--lr 5e-6 \
	--thresd 0.05 \
       	--logging_steps 50 \
	--k 3 \
       	--stride 16 \
	--max_passage_len 128 \
       	--filter_span  \
	--type_encoder squad2_electra_large  \
	--type_span_loss sum_log \
       	--type_rank_loss sum_log  \
	--do_rerank \
       	--add_topic \
	--use_title \
	--thresd 0.01 \
	--fp16 \
       	--results_dir /platform_tech/yuanzheng/GENE2E/entqa/run/mm$split/reader_results/


split="_t038_res"
export CUDA_VISIBLE_DEVICES=$1
python run_reader.py  \
	--init_model /platform_tech/yuanzheng/GENE2E/entqa/checkpoints/reader.pt \
	--model /platform_tech/yuanzheng/GENE2E/entqa/run/mm$split/reader.pt \
	--data_dir /platform_tech/yuanzheng/GENE2E/entqa/run/mm$split/reader_input/  \
       	--kb_dir /platform_tech/yuanzheng/GENE2E/entqa/medmention/kb$split/ \
	--C 64 \
       	--B 2 \
       	--L 180 \
       	--C_val 100 \
       	--gpus $1 \
      	--val_bsz 32 \
	--gradient_accumulation_steps 2 \
       	--warmup_proportion 0.06  \
	--epochs 30 \
       	--lr 5e-6 \
	--thresd 0.05 \
       	--logging_steps 50 \
	--k 3 \
       	--stride 16 \
	--max_passage_len 128 \
       	--filter_span  \
	--type_encoder squad2_electra_large  \
	--type_span_loss sum_log \
       	--type_rank_loss sum_log  \
	--do_rerank \
       	--add_topic \
	--use_title \
	--thresd 0.01 \
	--fp16 \
       	--results_dir /platform_tech/yuanzheng/GENE2E/entqa/run/mm$split/reader_results/


split="_t058_res"
export CUDA_VISIBLE_DEVICES=$1
python run_reader.py  \
	--init_model /platform_tech/yuanzheng/GENE2E/entqa/checkpoints/reader.pt \
	--model /platform_tech/yuanzheng/GENE2E/entqa/run/mm$split/reader.pt \
	--data_dir /platform_tech/yuanzheng/GENE2E/entqa/run/mm$split/reader_input/  \
       	--kb_dir /platform_tech/yuanzheng/GENE2E/entqa/medmention/kb$split/ \
	--C 64 \
       	--B 2 \
       	--L 180 \
       	--C_val 100 \
       	--gpus $1 \
      	--val_bsz 32 \
	--gradient_accumulation_steps 2 \
       	--warmup_proportion 0.06  \
	--epochs 30 \
       	--lr 5e-6 \
	--thresd 0.05 \
       	--logging_steps 50 \
	--k 3 \
       	--stride 16 \
	--max_passage_len 128 \
       	--filter_span  \
	--type_encoder squad2_electra_large  \
	--type_span_loss sum_log \
       	--type_rank_loss sum_log  \
	--do_rerank \
       	--add_topic \
	--use_title \
	--thresd 0.01 \
	--fp16 \
       	--results_dir /platform_tech/yuanzheng/GENE2E/entqa/run/mm$split/reader_results/
