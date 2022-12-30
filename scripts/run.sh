gpu=$1
tmp=$2
dataset=$3

CUDA_VISIBLE_DEVICES=${gpu} python3 gpt2_train.py --model_name_or_path gpt2-medium --train_file data/socialqa/train_gpt2_no_perm.csv --output_dir ${tmp}/gpt2-medium-no-trainer/ --per_device_train_batch_size 2 --gradient_accumulation_steps 512 --learning_rate 0.0005 --num_warmup_steps 100

CUDA_VISIBLE_DEVICES=${gpu} python3 gpt2_train.py --seed 13 --model_name_or_path ${tmp}/gpt2-medium-no-trainer/ --train_file data/sst/train_gpt2_no_perm_8_13.csv --output_dir ${tmp}/gpt2-medium-socialqa-8-seed-13/ --per_device_train_batch_size 2 --gradient_accumulation_steps 512 --learning_rate 0.0005 --num_warmup_steps 100
CUDA_VISIBLE_DEVICES=${gpu} python3 generate_dataset_sst.py --seed 13 --model_name_or_path ${tmp}/gpt2-medium-socialqa-8-seed-13/ --output_dir ${tmp} --strategy topk --num_tries 450
python3 data/sst/combine_train_gen_data_for_bert.py ${tmp} data/sst/train_8_13.pkl ${tmp}/df_gen_topk_450_sampling.pkl
CUDA_VISIBLE_DEVICES=${gpu} python3 train_classifier.py --early_stop 0 --seed 13 --do_predict --model_name_or_path bert-base-uncased --train_file ${tmp}/train_450_combined.csv --validation_file data/sst/val_8_13.csv --test_file data/sst/test.csv --max_length 512 --per_device_train_batch_size 32 --per_device_eval_batch_size 128 --num_train_epochs 4 --output_dir ${tmp}/bert > output/sst/bert_socialqa_gen_8_sup_seed_13.txt
rm ${tmp}/df_gen_topk_450_sampling.pkl
rm ${tmp}/train_450_combined.pkl
rm ${tmp}/df_gen_topk_450_sampling.csv
rm ${tmp}/train_450_combined.csv
CUDA_VISIBLE_DEVICES=${gpu} python3 gpt2_train.py --seed 21 --model_name_or_path ${tmp}/gpt2-medium-no-trainer/ --train_file data/sst/train_gpt2_no_perm_8_21.csv --output_dir ${tmp}/gpt2-medium-socialqa-8-seed-21/ --per_device_train_batch_size 2 --gradient_accumulation_steps 512 --learning_rate 0.0005 --num_warmup_steps 100
CUDA_VISIBLE_DEVICES=${gpu} python3 generate_dataset_sst.py --seed 21 --model_name_or_path ${tmp}/gpt2-medium-socialqa-8-seed-21/ --output_dir ${tmp} --strategy topk --num_tries 450
python3 data/sst/combine_train_gen_data_for_bert.py ${tmp} data/sst/train_8_21.pkl ${tmp}/df_gen_topk_450_sampling.pkl
CUDA_VISIBLE_DEVICES=${gpu} python3 train_classifier.py --early_stop 0 --seed 21 --do_predict --model_name_or_path bert-base-uncased --train_file ${tmp}/train_450_combined.csv --validation_file data/sst/val_8_21.csv --test_file data/sst/test.csv --max_length 512 --per_device_train_batch_size 32 --per_device_eval_batch_size 128 --num_train_epochs 4 --output_dir ${tmp}/bert > output/sst/bert_socialqa_gen_8_sup_seed_21.txt
rm ${tmp}/df_gen_topk_450_sampling.pkl
rm ${tmp}/train_450_combined.pkl
rm ${tmp}/df_gen_topk_450_sampling.csv
rm ${tmp}/train_450_combined.csv
CUDA_VISIBLE_DEVICES=${gpu} python3 gpt2_train.py --seed 42 --model_name_or_path ${tmp}/gpt2-medium-no-trainer/ --train_file data/sst/train_gpt2_no_perm_8_42.csv --output_dir ${tmp}/gpt2-medium-socialqa-8-seed-42/ --per_device_train_batch_size 2 --gradient_accumulation_steps 512 --learning_rate 0.0005 --num_warmup_steps 100
CUDA_VISIBLE_DEVICES=${gpu} python3 generate_dataset_sst.py --seed 42 --model_name_or_path ${tmp}/gpt2-medium-socialqa-8-seed-42/ --output_dir ${tmp} --strategy topk --num_tries 450
python3 data/sst/combine_train_gen_data_for_bert.py ${tmp} data/sst/train_8_42.pkl ${tmp}/df_gen_topk_450_sampling.pkl
CUDA_VISIBLE_DEVICES=${gpu} python3 train_classifier.py --early_stop 0 --seed 42 --do_predict --model_name_or_path bert-base-uncased --train_file ${tmp}/train_450_combined.csv --validation_file data/sst/val_8_42.csv --test_file data/sst/test.csv --max_length 512 --per_device_train_batch_size 32 --per_device_eval_batch_size 128 --num_train_epochs 4 --output_dir ${tmp}/bert > output/sst/bert_socialqa_gen_8_sup_seed_42.txt
rm ${tmp}/df_gen_topk_450_sampling.pkl
rm ${tmp}/train_450_combined.pkl
rm ${tmp}/df_gen_topk_450_sampling.csv
rm ${tmp}/train_450_combined.csv