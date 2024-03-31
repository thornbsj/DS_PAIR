python ./casual_predict.py --model_name=bigscience/bloomz-560m_32bit_False_1/checkpoint-500 --tokenizer_name=bigscience/bloomz-560m
python ./casual_predict.py --model_name=bigscience/bloomz-560m_32bit_False_3/checkpoint-1500 --tokenizer_name=bigscience/bloomz-560m
python ./casual_predict.py --model_name=facebook/xglm-564M_32bit_False_1/checkpoint-500 --tokenizer_name=facebook/xglm-564M --use_GPU=True
python ./casual_predict.py --model_name=facebook/xglm-564M_32bit_False_3/checkpoint-1500 --tokenizer_name=facebook/xglm-564M --use_GPU=True
python ./casual_predict.py --model_name=bigscience/bloom-7b1_4bit_True_1/checkpoint-500 --tokenizer_name=bigscience/bloom-7b1 --use_GPU=True