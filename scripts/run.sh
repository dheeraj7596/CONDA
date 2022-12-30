gpu=$1
tmp=$2
dataset=$3

CUDA_VISIBLE_DEVICES=${gpu} python3 gpt2_train.py --model_name_or_path gpt2-medium --train_file data/qa/socialiqa/train_qac.csv --output_dir ${tmp}/gpt2-medium-no-trainer/ --per_device_train_batch_size 2 --gradient_accumulation_steps 512 --learning_rate 0.0005 --num_warmup_steps 100

CUDA_VISIBLE_DEVICES=${gpu} python3 gpt2_train.py --seed 13 --model_name_or_path ${tmp}/gpt2-medium-no-trainer/ --train_file data/cls/${dataset}/train_qac_13.csv --output_dir ${tmp}/gpt2-medium-socialiqa-8-seed-13/ --per_device_train_batch_size 2 --gradient_accumulation_steps 512 --learning_rate 0.0005 --num_warmup_steps 100
CUDA_VISIBLE_DEVICES=${gpu} python3 generate_dataset.py --dataset_name ${dataset} --seed 13 --model_name_or_path ${tmp}/gpt2-medium-socialiqa-8-seed-13/ --output_dir ${tmp} --strategy topk --num_tries 450
python3 combine.py ${tmp} data/cls/${dataset}/train/train_13.csv ${tmp}/df_gen_topk_450_sampling.csv
CUDA_VISIBLE_DEVICES=${gpu} python3 train_classifier.py --early_stop 0 --seed 13 --do_predict --model_name_or_path bert-base-uncased --train_file ${tmp}/train_450_combined.csv --validation_file data/cls/${dataset}/val/val_13.csv --test_file data/cls/${dataset}/test/test.csv --max_length 512 --per_device_train_batch_size 32 --per_device_eval_batch_size 128 --num_train_epochs 4 --output_dir ${tmp}/bert
rm ${tmp}/df_gen_topk_450_sampling.csv
rm ${tmp}/train_450_combined.csv
CUDA_VISIBLE_DEVICES=${gpu} python3 gpt2_train.py --seed 21 --model_name_or_path ${tmp}/gpt2-medium-no-trainer/ --train_file data/cls/${dataset}/train_qac_21.csv --output_dir ${tmp}/gpt2-medium-socialiqa-8-seed-21/ --per_device_train_batch_size 2 --gradient_accumulation_steps 512 --learning_rate 0.0005 --num_warmup_steps 100
CUDA_VISIBLE_DEVICES=${gpu} python3 generate_dataset.py --dataset_name ${dataset} --seed 21 --model_name_or_path ${tmp}/gpt2-medium-socialiqa-8-seed-21/ --output_dir ${tmp} --strategy topk --num_tries 450
python3 combine.py ${tmp} data/cls/${dataset}/train/train_21.csv ${tmp}/df_gen_topk_450_sampling.csv
CUDA_VISIBLE_DEVICES=${gpu} python3 train_classifier.py --early_stop 0 --seed 21 --do_predict --model_name_or_path bert-base-uncased --train_file ${tmp}/train_450_combined.csv --validation_file data/cls/${dataset}/val/val_21.csv --test_file data/cls/${dataset}/test/test.csv --max_length 512 --per_device_train_batch_size 32 --per_device_eval_batch_size 128 --num_train_epochs 4 --output_dir ${tmp}/bert
rm ${tmp}/df_gen_topk_450_sampling.csv
rm ${tmp}/train_450_combined.csv
CUDA_VISIBLE_DEVICES=${gpu} python3 gpt2_train.py --seed 42 --model_name_or_path ${tmp}/gpt2-medium-no-trainer/ --train_file data/cls/${dataset}/train_qac_42.csv --output_dir ${tmp}/gpt2-medium-socialiqa-8-seed-42/ --per_device_train_batch_size 2 --gradient_accumulation_steps 512 --learning_rate 0.0005 --num_warmup_steps 100
CUDA_VISIBLE_DEVICES=${gpu} python3 generate_dataset.py --dataset_name ${dataset} --seed 42 --model_name_or_path ${tmp}/gpt2-medium-socialiqa-8-seed-42/ --output_dir ${tmp} --strategy topk --num_tries 450
python3 combine.py ${tmp} data/cls/${dataset}/train/train_42.csv ${tmp}/df_gen_topk_450_sampling.csv
CUDA_VISIBLE_DEVICES=${gpu} python3 train_classifier.py --early_stop 0 --seed 42 --do_predict --model_name_or_path bert-base-uncased --train_file ${tmp}/train_450_combined.csv --validation_file data/cls/${dataset}/val/val_42.csv --test_file data/cls/${dataset}/test/test.csv --max_length 512 --per_device_train_batch_size 32 --per_device_eval_batch_size 128 --num_train_epochs 4 --output_dir ${tmp}/bert
rm ${tmp}/df_gen_topk_450_sampling.csv
rm ${tmp}/train_450_combined.csv