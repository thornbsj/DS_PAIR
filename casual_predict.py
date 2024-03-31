import argparse
from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer,AutoModelForCausalLM,DataCollatorForLanguageModeling,TrainingArguments,Trainer
import torch
from peft import LoraConfig, TaskType, get_peft_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DSPairGen')
    tests = pd.read_excel("For_generation.xlsx")
    parser.add_argument('--model_name', type=str, default='bigscience/bloomz-560m',
                            help='fine-tuned model name')
        
    parser.add_argument('--tokenizer_name', type=str, default='bigscience/bloomz-560m',
                        help='tokenizer name from huggingface')
    
    parser.add_argument('--use_GPU', type=bool, default=False,
                            help='if generate on GPU')
    
    parser.add_argument('--do_sample', type=bool, default=False,
                            help='if do sample while generation')
    
    parser.add_argument('--no_repeat_ngram_size', type=int, default=1,
                            help='limit repeat n_gram')
    
    parser.add_argument('--num_beams', type=int, default=1,
                            help='beam numbers of beam search')
    
    parser.add_argument('--model_load_type', type=str, default='32bit',
                            help='if you need to load model in 16bit/8bit/4bit(qLora)...')
    
    parser.add_argument('--max_length', type=int, default=150,
                            help='max_length to predict')
    args = parser.parse_args()

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

    model = AutoModelForCausalLM.from_pretrained(args.model_name,**model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if args.use_GPU and args.model_load_type not in ("4bit","8bit"):
        model = model.cuda()

    res = []
    for i in tests["des_above"]:
        ipt = tokenizer(i,return_tensors="pt").to(model.device)
        ipt["do_sample"] = args.do_sample
        ipt["no_repeat_ngram_size"] = args.no_repeat_ngram_size
        if args.num_beams>1:
            ipt["num_beams"] = args.num_beams
        res.append(tokenizer.decode(model.generate(**ipt,max_length=args.max_length)[0]))
    tests["res"]=res
    tests.to_excel(args.model_name+'/result.xlsx')