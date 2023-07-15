

# # # training kebiolm
# # cd ./KeBioLM/ner
# # KEBIOLM_CHECKPOINT_PATH=/platform_tech/yuanzheng/GENE2E/kebiolm/
# # BC5CDR_DATASET=/platform_tech/yuanzheng/GENE2E/MM_dataset/all_kb_kebiolm/
# # OUTPUT_DIR=/platform_tech/yuanzheng/GENE2E/MM_dataset/all_kb_kebiolm/model_saved/

# # CUDA_VISIBLE_DEVICES=$1 python \
# # run_ner.py \
# # --data_dir $BC5CDR_DATASET \
# # --model_name_or_path $KEBIOLM_CHECKPOINT_PATH \
# # --output_dir $OUTPUT_DIR \
# # --tokenizer_name /platform_tech/yuanzheng/GENE2E/kebiolm/ \
# # --do_train --do_eval --num_train_epochs 20 \
# # --do_predict --overwrite_output_dir \
# # --gradient_accumulation_steps 2 \
# # --learning_rate 1e-5 \
# # --warmup_steps 570 \
# # --evaluation_strategy epoch \
# # --max_seq_length 512 \
# # --per_device_train_batch_size 8 \
# # --eval_accumulation_steps 1 \
# # --load_best_model_at_end --metric_for_best_model f1


# ## prediction endtoend linking


# cd ./KeBioLM/ner

# KEBIOLM_CHECKPOINT_PATH=/platform_tech/yuanzheng/GENE2E/BC5CDR_mesh_kebiolm/checkpoint-980
# BC5CDR_DATASET=/platform_tech/yuanzheng/GENE2E/BC5CDR_mesh_kebiolm/
# OUTPUT_DIR=/platform_tech/yuanzheng/GENE2E/BC5CDR_mesh_kebiolm/mesh_train_result

# # KEBIOLM_CHECKPOINT_PATH=/platform_tech/yuanzheng/GENE2E/MM_dataset/all_kb_kebiolm/model_saved/checkpoint-9320
# # BC5CDR_DATASET=/platform_tech/yuanzheng/GENE2E/MM_dataset/subset_kb_kebiolm_snomed/
# # OUTPUT_DIR=/platform_tech/yuanzheng/GENE2E/MM_dataset/all_kb_kebiolm/all_train_result

# CUDA_VISIBLE_DEVICES=$1 python \
# run_ner.py \
# --data_dir $BC5CDR_DATASET \
# --model_name_or_path $KEBIOLM_CHECKPOINT_PATH \
# --output_dir $OUTPUT_DIR \
# --tokenizer_name /platform_tech/yuanzheng/GENE2E/kebiolm/ \
# --num_train_epochs 20 \
# --do_predict --overwrite_output_dir \
# --gradient_accumulation_steps 2 \
# --learning_rate 1e-5 \
# --warmup_steps 570 \
# --evaluation_strategy epoch \
# --max_seq_length 512 \
# --per_device_train_batch_size 8 \
# --eval_accumulation_steps 1 \
# --load_best_model_at_end --metric_for_best_model f1


# cd ../../

# CUDA_VISIBLE_DEVICES=$1 python ./main_coder_mm.py \
#     --model_path GanjinZero/UMLSBert_ENG \
#     --train_kb all \
#     --test_kb all \
#     --test_annot all \
#     --test_set test \
#     --subkb T038 \


# CUDA_VISIBLE_DEVICES=2 python ./main_coder_mm.py \
#     --model_path GanjinZero/UMLSBert_ENG \
#     --train_kb all \
#     --test_kb all \
#     --test_annot diffset \
#     --test_set test \
#     --subkb T038 \

# CUDA_VISIBLE_DEVICES=2 python ./main_coder_mm.py \
#     --model_path GanjinZero/UMLSBert_ENG \
#     --train_kb all \
#     --test_kb all \
#     --test_annot subset \
#     --test_set test \
#     --subkb T038 \

# CUDA_VISIBLE_DEVICES=2 python ./main_coder_mm.py \
#     --model_path GanjinZero/UMLSBert_ENG \
#     --train_kb all \
#     --test_kb all \
#     --test_annot diffset \
#     --test_set test \
#     --subkb T058 \

# CUDA_VISIBLE_DEVICES=2 python ./main_coder_mm.py \
#     --model_path GanjinZero/UMLSBert_ENG \
#     --train_kb all \
#     --test_kb all \
#     --test_annot subset \
#     --test_set test \
#     --subkb T058 \

# CUDA_VISIBLE_DEVICES=2 python ./main_coder_mm.py \
#     --model_path GanjinZero/UMLSBert_ENG \
#     --train_kb all \
#     --test_kb all \
#     --test_annot diffset \
#     --test_set test \
#     --subkb snomed \

# CUDA_VISIBLE_DEVICES=2 python ./main_coder_mm.py \
#     --model_path GanjinZero/UMLSBert_ENG \
#     --train_kb all \
#     --test_kb all \
#     --test_annot subset \
#     --test_set test \
#     --subkb snomed \

# CUDA_VISIBLE_DEVICES=1 python ./main_coder.py \
#     --model_path GanjinZero/UMLSBert_ENG \
#     --train_kb all \
#     --test_kb all \
#     --test_annot diffset \
#     --test_set test \
#     # --subkb snomed \

# CUDA_VISIBLE_DEVICES=1 python ./main_coder.py \
#     --model_path GanjinZero/UMLSBert_ENG \
#     --train_kb all \
#     --test_kb all \
#     --test_annot subset \
#     --test_set test \
#     # --subkb snomed \

# CUDA_VISIBLE_DEVICES=1 python genre_eval.py all all subset test mm T038

# CUDA_VISIBLE_DEVICES=1 python genre_eval.py all all subset test mm T058
# CUDA_VISIBLE_DEVICES=1 python genre_eval.py all all diffset test mm T058

# CUDA_VISIBLE_DEVICES=1 python genre_eval.py all all subset test mm snomed
# CUDA_VISIBLE_DEVICES=1 python genre_eval.py all all diffset test mm snomed

CUDA_VISIBLE_DEVICES=1 python genre_eval.py all all subset test bc5cdr medic
CUDA_VISIBLE_DEVICES=1 python genre_eval.py all all diffset test bc5cdr medic