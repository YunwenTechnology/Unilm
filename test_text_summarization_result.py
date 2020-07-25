"""
@author: liucong
@contact: logcongcong@gmail.com
@time: 2020/7/23 15:28
"""

from rouge import FilesRouge, Rouge
import nltk


hyp_path = "/data/unilm/text_summarization_data/1.txt"
ref_path = "/data/unilm/text_summarization_data/1.txt"
hyp_str = "卧槽.碉堡了威武"
ref_str = " 卧槽.碉堡了！ "
files_rouge = FilesRouge()
scores = files_rouge.get_scores(hyp_path, ref_path, avg=True)
print(scores)