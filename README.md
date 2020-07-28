# Unilm
## 简介
UniLM模型既可以应用于自然语言理解（NLU）任务，又可以应用于自然语言生成（NLG）任务。[论文](https://arxiv.org/abs/1905.03197)来自微软研究院。

模型虽然强大，但微软并没有开源中文的预训练模型。因此云问本着开源之前，将我们预训练好的中文unilm_base模型进行开源。
## 模型链接
| | 模型地址  | 提取码 |
| ------------- | ------------- | ------------- |
| tf | [百度云盘](https://pan.baidu.com/s/1HgxIkBl5Yfwrzs1K1B6NFA)  | tblr |
| torch | [百度云盘](https://pan.baidu.com/s/1DHJGOFJ5cce5N5g4aBDiMQ) | etwf |
## 详细介绍
详细介绍见：[知乎链接](https://zhuanlan.zhihu.com/p/163483660)

在CLUE中的部分分类数据集和阅读理解数据集上进行了简单测试，具体效果如下：

| 模型  | AFQMC | TNEWS | IFLYTEK | CMNLI | CSL | CMRC2018 | AVG |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| bert_base | 73.70% | 56.58% | 60.29% | 79.69% | 80.36% | 71.60% | 70.37% |
| ERNIE_base | 73.83% | 58.33% | 58.96% | 80.29% | 79.10% | 74.70% | 70.87% |
| unilm_base | 73.79% | 56.27% | 60.54% | 79.58% | 80.80% | 73.30% | 70.71% |


对CLUE中[新闻摘要数据](https://www.cluebenchmarks.com/dataSet_search_modify.html?keywords=%E6%96%87%E6%9C%AC%E6%91%98%E8%A6%81)，训练了摘要模型，并进行的简单的测试，具体效果如下：

| 模型  | rouge-1 | rouge-2 | rouge-L |
| ------------- | ------------- | ------------- | ------------- |
| f1 | 43.98% | 32.04% | 49.90% |
| r | 41.74% | 30.50% | 47.35% |
| p | 49.30% | 35.85% | 56.01% |

如何使用该模型进行NLU任务？
* 可以直接使用BERT代码，只需加载unilm的model、config、vocab即可。

如何使用该模型进行NLG任务？
* fine-tuning
~~~
nohup python3 -u run_seq2seq.py --data_dir /data/unilm/data_file/ --src_file train_data.json --model_type unilm --model_name_or_path /data/unilm/yunwen_unilm/ --output_dir /data/unilm/output_dir/ --max_seq_length 512 --max_position_embeddings 512 --do_train --do_lower_case --train_batch_size 32 --learning_rate 1e-5 --num_train_epochs 3 > log.log 2>&1 &
~~~
* test
~~~
python3 -u decode_seq2seq.py --model_type unilm --model_name_or_path /data/unilm/yunwen_unilm/ --model_recover_path /data/unilm/output_dir/model.bin --max_seq_length 512 --input_file /data/unilm/data_file/test.json --output_file /data/unilm/data_file/predict_.json --do_lower_case --batch_size 32 --beam_szie 5 --max_tgt_length 128
~~~

注：根据论文，在NLU任务时，type_token_id为[0,1]；在NLG任务时，type_token_id为[4,5]

补充摘要模型对比实验（参考：[CLGE](https://github.com/CLUEbenchmark/CLGE)）
* 数据集CSL 中长文本摘要生成

[百度网盘](https://pan.baidu.com/s/1-KfE5oXMJE8Ia2npNxj9fw) 提取码：y2zj

| 模型  | rouge-1 | rouge-2 | rouge-L | BLUE | 参数 |
| ------------- | ------------- | ------------- | ------------- |  ------------- |  ------------- |
| bert_base | 61.71% | 50.97% | 60.51% | 41.10% | batch_size=24, length=512, epoch=5, lr=1e-5 |
| unilm | 62.13% | 51.20% | 60.61% | 41.81% |  batch_size=24, length=512, epoch=5, lr=1e-5  |


* 微博新闻摘要数据，从[新闻摘要数据](https://www.cluebenchmarks.com/dataSet_search_modify.html?keywords=%E6%96%87%E6%9C%AC%E6%91%98%E8%A6%81)中随机挑选10000篇作为训练集，1000篇作为测试集。

[百度网盘](https://pan.baidu.com/s/1Vl6Qb7eOEc64oygsC_ec8Q) 提取码：htmh

| 模型  | rouge-1 | rouge-2 | rouge-L | BLUE | 参数 |
| ------------- | ------------- | ------------- | ------------- |  ------------- |  ------------- |
| bert_base | 39.74% | 28.69% | 38.68% | 20.02% | batch_size=24, length=512, epoch=5, lr=1e-5 |
| unilm | 40.58% | 29.60% | 39.21% | 21.35% |  batch_size=24, length=512, epoch=5, lr=1e-5  |

# 训练环境
* torch 1.4.0
* transformers 2.6.0

# 联系我们
cliu@iyunwen.com

# 相关链接
http://www.iyunwen.com/