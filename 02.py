# 使用 r 表示 raw string，即不转义反斜杠
# encoding="utf-8" 表示以 utf-8 编码读取文件
with open(r"the-verdict.txt", "r", encoding="utf-8") as f:
     raw_text = f.read()
print("total number of characters in the text:", len(raw_text))
print(raw_text[:1000])

# import os
# print(os.getcwd())
# 注意：os.getcwd() 返回的是当前工作目录，即当前 Python 脚本运行的目录。
# 如果你在命令行下运行 Python 脚本，则当前工作目录就是你当前打开的命令行窗口所在目录。
# 如果你在 IDE 里运行 Python 脚本，则当前工作目录就是 IDE 的工作目录。
# 所以请注意，我们应该上面的 open 里面的文件路径是相对于当前工作目录的，而不是相对于脚本所在目录的。

import re

preprocessed = re.split(r'([,.?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]

# print(len(preprocessed))

all_words = sorted(list(set(preprocessed)))
vocab_size = len(all_words)
print("vocab size:", vocab_size)

# vocab = {token: integer for integer, token in enumerate(all_words)}
# for i, item in enumerate(vocab.items()):
#      print(item)
#      if i > 50:
#           break

# 实现一个简单的文本标记器
class SimpleTokenizerV1:
    # 构建函数，用于给定词汇表初始化分词器
    def __init__(self, vocab):
        # 'vocab' 是一个词汇表，是一个字典，键是字符串，值是整数
        self.str_to_int = vocab
        # 使用字典表推导式创建一个从整数到字符串的反向映射
        self.int_to_str = {i: s for s, i in vocab.items()}
        
    def encode(self, text):
        # 将给定的文本编码为基于词汇表的整数列表。
        
        # 使用正则表达式对文本进行预处理和分词。
        # 正则表达式根据标点符号和空格进行分割，并在结果中保留这些字符。
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)

        # 删除预处理列表中的任何空白或仅包含空格的项。
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        
        # 使用词汇表将每个标记（字符串）转换为对应的整数ID。
        ids = [self.str_to_int[s] for token in preprocessed]
        return ids

    def decode(self, ids):
        # 使用反向词汇表将整数列表解码回字符串。
        
        # 将整数列表转换回字符串并用空格连接。
        text = " ".join([self.int_to_str[i] for i in ids])
        # 使用正则表达式去除标点符号前多余的空格。
        text = re.sub(r'\s+([,.?_!"()\'])', r'\1', text)
        return text

class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.str_to_int  # A
                        else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)  # B
        return text
    

from importlib.metadata import version
import tiktoken
print("tiktoken version:", version("tiktoken"))




     
max_length = 4
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_length, stride=stride
)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Token IDs:\n", inputs)
print("\nInputs shape:\n", inputs.shape)