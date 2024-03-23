python ./fine_tune.py --model_name=google/mt5-small --tokenizer_name=google/mt5-small --per_device_train_batch_size=4 --gradient_accumulation_steps=8
python ./fine_tune.py --model_name=fnlp/bart-base-chinese --tokenizer_name=fnlp/bart-base-chinese --per_device_train_batch_size=4 --gradient_accumulation_steps=8
python ./fine_tune.py --model_name=Langboat/mengzi-t5-base --tokenizer_name=Langboat/mengzi-t5-base --per_device_train_batch_size=4 --gradient_accumulation_steps=8
python ./fine_tune.py --model_name=facebook/m2m100_418M --tokenizer_name=facebook/m2m100_418M --per_device_train_batch_size=4 --gradient_accumulation_steps=8
python ./fine_tune.py --model_name=IDEA-CCNL/Randeng-T5-784M-MultiTask-Chinese --tokenizer_name=IDEA-CCNL/Randeng-T5-784M-MultiTask-Chinese  --per_device_train_batch_size=4 --gradient_accumulation_steps=8
python ./fine_tune.py --model_name=mxmax/Chinese_Chat_T5_Base --tokenizer_name=mxmax/Chinese_Chat_T5_Base --per_device_train_batch_size=4 --gradient_accumulation_steps=8
