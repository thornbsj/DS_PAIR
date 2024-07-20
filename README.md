# 根据物品的描述以及效果生成“黑暗之魂”式的中文物品背景描述
数据来源：https://github.com/LeMinerva/Dark-Souls-Documents
<br>
我训练出的模型网盘：https://pan.baidu.com/s/1B16RGDCA_mNGaYafKaHPyQ?pwd=dxei 
提取码：dxei
使用模型：bigscience/bloomz-560m	
<br>
get_data_for_casualLM.ipynb：从上面的数据中摘取数据<br>
data_for_casual.xlsx：原始数据集<br>
Data Augumentation.ipynb：对数据进行数据增强<br>
augumentation.xlsx：增强后的数据集<br>
casual_train.py：使用增强后的数据集进行模型训练<br>
可选参数：<br>
  --model_name：从hugging-face上要下载用的模型名称<br>
  --tokenizer_name：从hugging-face上要下载用的文本tokenizer名称，通常和model_name相同<br>
  --model_load_type：是否要调整模型精度为16bit/8bit/4bit<br>
  --use_lora：是否要使用LoRa方法进行微调<br>
  --lora_r：LoRa中的秩数<br>
  --lora_drop：LoRa的dropout率<br>
  --target_modules：要对哪些部分进行LoRa微调，应该输入一个正则表达式。默认为"transformer.*query_key_value"<br>
  --num_epochs：模型微调轮数（通常设为1即可）<br>
  --per_device_train_batch_size、gradient_accumulation_steps：GPU上每次正向传播多少样本，样本累计多少轮才做反向传播。<br>
  --save_total_limit：最终要保存多少个模型（checkpoint）<br>
  --train_min_length：文本长度至少要达到多少才能进入训练集<br>
  --learning_rate：学习率，默认1e-5<br>
<br>
<br>
For_generation.xlsx：要做文本生成的上文，自己定义<br>
casual_predict.py：文本生成<br>
可选参数：<br>
  --model_name：自己训练的模型路径<br>
  --tokenizer_name：从hugging-face上要下载用的文本tokenizer名称，通常和model_name相同<br>
  --use_GPU：是否在GPU上进行推理<br>
  --do_sample：文本生成时是否采用“抽样”策略<br>
  --no_repeat_ngram_size：文本生成中限制“不要让词连续出现n次”，默认为1<br>
  --num_beams：文本生成时是否采用“波束搜索”策略，并限定波束数量，默认为1（大于1时才是波束搜索）。<br>
  --max_length：模型输出文本的最大长度（包括eos token）。<br>
<br>
可用train_casual_model.sh与predict_casual_model.sh作参考
<br>
2024年7月20日更新：<br>
新增文件夹DS_Desc；将Hugging-face训练完的模型放入dot NET项目内<br>
详细信息请看：https://blog.csdn.net/thorn_r/article/details/140576460<br>
注意文件夹中的Tokenizer文件在上传到github时可能有损坏了，完整的ONNX模型以及分词文件可以从此处下载：<br>
链接：https://pan.baidu.com/s/1Zl8Uj7J8ZYVj3lMtNpndAQ?pwd=q7tl<br> 
提取码：q7tl