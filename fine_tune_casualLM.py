import argparse
from datasets import Dataset,load_dataset
from transformers import AutoTokenizer,AutoModelForCausalLM,DataCollatorForSeq2Seq,TrainingArguments,EarlyStoppingCallback,Trainer
import pandas as pd
import torch
import numpy as np
from peft import LoraConfig, TaskType, get_peft_model

def getLoraConfig(target_modules=["q","k"],lora_dropout=0.05,r=8):
    lora_config = LoraConfig(
        r=r,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        task_type=TaskType.CAUSAL_LM
    )
    return lora_config

if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description='DSPairGen')
        #read_excel
        dataset = pd.read_excel("name_remark_description.xlsx")


        parser.add_argument('--model_name', type=str, required=True, default='Langboat/mengzi-t5-base',
                            help='model name from huggingface')
        
        parser.add_argument('--tokenizer_name', type=str, required=True, default='Langboat/mengzi-t5-base',
                            help='tokenizer name from huggingface')
        
        parser.add_argument('--model_load_type', type=str, default='32bit',
                            help='if you need to load data in 16bit/8bit/4bit(qLora)...')

        parser.add_argument('--use_lora', type=bool, default=False,
                            help='if you need to use LoRa (will be forced to use when load with 8bit/4bit)')
        
        parser.add_argument('--lora_r', type=int, default=8,
                            help='the parameter rank of LoRa')
        
        parser.add_argument('--lora_drop', type=float, default=0.05,
                            help='dropout parameter of LoRa')
        
        parser.add_argument('--target_modules', type=list, nargs='+', default=["q","k"],
                            help='target parameters of LoRa')
        
        parser.add_argument('--num_epochs', type=int, default=100,
                            help='maximum epoch of training')

        parser.add_argument('--per_device_train_batch_size', type=int, default=4,
                            help='per device train batch size')

        parser.add_argument('--gradient_accumulation_steps', type=int, default=8,
                            help='per_device_train_batch_size*gradient_accumulation_steps=one BP batch of training')

        parser.add_argument('--save_total_limit', type=int, default=1,
                            help='total save models')
        
        parser.add_argument('--eval_metric', type=str, default="rouge",
                            help='total save models')
        
        parser.add_argument('--learning_rate', type=float, default=1e-5)
        parser.add_argument('--early_stopping_patience', type=int, default=5)

        parser.add_argument('--prompt_string', type=str, default="请根据以下信息撰写出一个符合“黑暗之魂”系列风格的物品描述：",
                            help='total save models')

        args = parser.parse_args()
        print(args)
        #model_names
        model_name = args.model_name
        tokenizer_name = args.tokenizer_name

        model_param_dict = {
            "16bit":[("torch_dtype",torch.half)],
            "8bit":[("torch_dtype",torch.half),("load_in_8bit",True)],
            "4bit":[("torch_dtype",torch.half),("load_in_4bit",True),("bnb_4bit_compute_dtype",torch.half),("bnb_4bit_quant_type","nf4"),("bnb_4bit_use_double_quant",True)]
        }
        
        model_kwargs = {}
        

        model_load_type = args.model_load_type
        if model_load_type in model_param_dict.keys():
            for k,v in model_param_dict[model_load_type]:
                model_kwargs[k] = v
        
        model = AutoModelForCausalLM.from_pretrained(model_name,**model_kwargs)
        model.enable_input_require_grads()
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)


        dataset = Dataset.from_pandas(dataset)
        dataset = dataset.train_test_split(test_size=90)
        def preprocess_func(example):
            inputs = tokenizer("\n".join([
                args.prompt_string
                ,"物品类型："+example["type"]
                ,"子类型："+example["sub_type"]
                ,"物品名称："+example["name"]
                ,"物品功能："+example["remark"]
            ]),max_length=50,truncation=True)
            outputs = tokenizer(text_target=example["description"],max_length=256,truncation=True)
            inputs["labels"]=outputs["input_ids"]
            
            return inputs
        
        tokenized_data = dataset["train"].map(preprocess_func,remove_columns=dataset["train"].column_names)
        tokenized_eval = dataset["test"].map(preprocess_func,remove_columns=dataset["train"].column_names)
        if args.model_load_type in ("8bit","16bit"):
            args.use_lora = True
        if args.use_lora:
            lora_config = getLoraConfig(target_modules=args.target_modules,lora_dropout=args.lora_drop,r=args.lora_r)
            model = get_peft_model(model,lora_config)
            model.print_trainable_parameters()


        train_args = TrainingArguments(
            output_dir=f"{model_name}_{model_load_type}_{args.use_lora}_{args.eval_metric}",
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            gradient_checkpointing=True,
            logging_strategy="epoch",
            num_train_epochs=args.num_epochs,
            save_strategy="epoch",
            per_device_eval_batch_size=1,
            eval_accumulation_steps=2,
            metric_for_best_model="bleu" if args.eval_metric == "bleu" else "rouge-l",
            save_total_limit=args.save_total_limit,
            evaluation_strategy="epoch",
            load_best_model_at_end=True,
            predict_with_generate=True,
            learning_rate=args.learning_rate
        )

        trainer = Trainer(model=model,args=train_args,train_dataset=tokenized_data,
                data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer),
                tokenizer=tokenizer,
                eval_dataset = tokenized_eval,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)])
        
        trainer.train()
    except Exception as e:
        print(e)