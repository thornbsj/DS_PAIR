import argparse
import pandas as pd
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM,DataCollatorForSeq2Seq,TrainingArguments,Seq2SeqTrainer,Seq2SeqTrainingArguments,EarlyStoppingCallback
from datasets import Dataset
parser = argparse.ArgumentParser(description='batch_output_des')
parser.add_argument('--excel_dir',type=str,required=True)
#parser.add_argument('--output_dir',type=str,required=True)
parser.add_argument("--pretrained_model",type=str,required=True)

if __name__ == "__main__":
    args = parser.parse_args()
    dataset_excel = pd.read_excel(args.excel_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.pretrained_model)
    def preprocess_func(example):
        inputs = tokenizer(
            "物品类型：{}\n子类型：{}\n物品名称：{}\n物品描述：{}".format(example["type"],example["sub_type"],example["name"],example["remark"]).strip()
            ,return_tensors="pt")["input_ids"]
        
        return inputs
    
    dataset = Dataset.from_pandas(dataset_excel)
    tokenized_data = dataset.map(preprocess_func,remove_columns=dataset.column_names)

    tokenizer.decode(model.generate(tokenized_data,max_length=100))