import numpy as np
import os
import openai

openai.api_key = "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"

response = openai.Completion.create(
    model="text-davinci-003",  # 模型选择达芬奇
    prompt="基于pytorh实现focalloss损失函数",  # 提问
    temperature=0.8,
    max_tokens=2000,  # 生成答案的字节数
    top_p=1.0,  # 跟temperature有点类似，结果概率的前面的选择
    frequency_penalty=0.5,  # [-2,2]频率太高的词的惩罚,就是减少重复的词出现(比如小于0会出现很多重复词)
    presence_penalty=0.0,  # [-2,2]围绕着提问来回答的程度(比如小于0的回答会过于紧扣主题)
)

print(response.choices[0].text)