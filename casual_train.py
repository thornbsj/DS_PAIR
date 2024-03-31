import argparse
from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer,AutoModelForCausalLM,DataCollatorForLanguageModeling,TrainingArguments,Trainer
import torch
from peft import LoraConfig, TaskType, get_peft_model

def getLoraConfig(target_modules=["q","k"],lora_dropout=0.0,r=8):
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
        data = pd.read_excel("augumentation.xlsx")
        ds = Dataset.from_pandas(data)
        parser.add_argument('--model_name', type=str, default='bigscience/bloomz-560m',
                            help='model name from huggingface')
        
        parser.add_argument('--tokenizer_name', type=str, default='bigscience/bloomz-560m',
                            help='tokenizer name from huggingface')
        
        parser.add_argument('--model_load_type', type=str, default='32bit',
                            help='if you need to load model in 16bit/8bit/4bit(qLora)...')

        parser.add_argument('--use_lora', type=bool, default=False,
                            help='if you need to use LoRa (will be forced to use when load with 8bit/4bit)')
        
        parser.add_argument('--lora_r', type=int, default=8,
                            help='the parameter rank of LoRa')
        
        parser.add_argument('--lora_drop', type=float, default=0.0,
                            help='dropout parameter of LoRa')
        
        parser.add_argument('--target_modules', type=str, default="transformer.*query_key_value",
                            help='target parameters of LoRa')
        
        parser.add_argument('--num_epochs', type=int, default=1,
                            help='epoch of training')

        parser.add_argument('--per_device_train_batch_size', type=int, default=4,
                            help='per device train batch size')

        parser.add_argument('--gradient_accumulation_steps', type=int, default=8,
                            help='per_device_train_batch_size*gradient_accumulation_steps=one BP batch of training')

        parser.add_argument('--save_total_limit', type=int, default=1,
                            help='total save models')
        
        parser.add_argument('--train_min_length', type=int, default=50,
                            help='min length of sentences in training set')
        
        parser.add_argument('--learning_rate', type=float, default=1e-5)
        args = parser.parse_args()
        print(args)
        
        ds = ds.select([i for i,x in enumerate(data["description"]) if len(x)>=args.train_min_length])

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

        def preprocess(examples):
            if tokenizer.eos_token is not None:
                contents = [e + tokenizer.eos_token for e in examples["description"]]
            else:
                contents = examples["description"]
            return tokenizer(contents,max_length = 200)

        tokenized_ds = ds.map(preprocess,batched=True,remove_columns = ds.column_names)
        if args.model_load_type in ("8bit","4bit"):
            args.use_lora = True
        if args.use_lora:
            lora_config = getLoraConfig(target_modules=args.target_modules,lora_dropout=args.lora_drop,r=args.lora_r)
            model = get_peft_model(model,lora_config)
            model.print_trainable_parameters()

        train_args = TrainingArguments(
            output_dir=f"{model_name}_{model_load_type}_{args.use_lora}_{args.num_epochs}",
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            logging_steps=10,
            num_train_epochs=args.num_epochs,
            gradient_checkpointing=True,
            save_total_limit=args.save_total_limit,
            learning_rate=args.learning_rate
        )
        trainer = Trainer(
            args = train_args,
            model = model,
            train_dataset = tokenized_ds,
            data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=False)
        )
        trainer.train()
    except Exception as e:
        print(e)