# 玩转各类语言模型

在当今各种语言模型爆发的年代, 有时候感觉跟紧最新的进展就已经是不容易的事情, 尽管ChatGPT出来已经四五个月了, 最近的一个月内, 各类的大语言模型频出不穷, 每个语言模型都有自己的技术和特征。









### ChatYuan

yuan是支持中英双语的大模型, 金主是浪潮, [ChatYuan](https://github.com/clue-ai/ChatYuan))应该就是基于Yuan训练的。

```
git clone git@github.com:clue-ai/ChatYuan.git
```



### Vicuna和Chinese-Vicuna, FastChat

[FastChat](https://github.com/lm-sys/FastChat)这是一个开源的可以支持训练, 本地部署, 衡量聊天语言模型的框架。其中release的语言模型为[**Vicuna**](https://vicuna.lmsys.org/)。团队来自UC伯克利, CMU, 斯坦福, UCSD。

```
git clone git@github.com:lm-sys/FastChat.git
git clone git@github.com:Facico/Chinese-Vicuna.git
```





### MiniGPT-4

[MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4)来自阿卜杜拉国王科技大学(King Abdullah University of Science and Technology), 

```
git clone git@github.com:Vision-CAIR/MiniGPT-4.git
```





### ColossalChat

 [ColossalChat](https://github.com/hpcaitech/ColossalAI/tree/main/applications/Chat), 来自ColossalAI。

```
git clone git@github.com:hpcaitech/ColossalAI.git
```



### ChatGLM-6B

[ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B), 是清华技术成果转化的公司智谱AI开源的GLM系列的对话模型，支持中英两个语种，目前开源了其62亿参数量的模型。其继承了GLM之前的优势，在模型架构上进行了优化，从而使得部署和应用门槛变低，实现大模型在消费级显卡上的推理应用。

```
git clone https://github.com/THUDM/ChatGLM-6B
```





### OpenFlamingo

[ OpenFlamingo](https://github.com/mlfoundations/open_flamingo)是一个对标GPT-4、支持大型多模态模型训练和评估的框架，由非盈利机构LAION重磅开源发布，其是对DeepMind的Flamingo模型的复现。目前开源的是其基于LLaMA的 OpenFlamingo-9B模型。Flamingo模型在包含交错文本和图像的大规模网络语料库上进行训练，具备上下文少样本学习能力。OpenFlamingo实现了原始Flamingo中提出的相同架构，在一个新的多模态C4数据集的5M样本和LAION-2B的10M样本上训练而来。

```
git clone git@github.com:mlfoundations/open_flamingo.git
```





### ChatLLaMA

 ChatLLaMA是由[Nebuly+AI](https://github.com/nebuly-ai/nebullvm)推出的基于人类反馈强化学习的LLaMA+AI聊天机器人的开源实现，它的技术路线类似 ChatGPT。

```
git clone git@github.com:nebuly-ai/nebullvm.git
```





### alpaca

[alpaca](https://github.com/tatsu-lab/stanford_alpaca)由斯坦福大学发布, 。

```
git clone git@github.com:tatsu-lab/stanford_alpaca.git
```





### OpenChatKit

OpenChatKit由前OpenAI研究员所在的Together团队，以及LAION、Ontocord.ai团队共同打造。OpenChatKit包含200亿个参数，用GPT-3的开源版本GPT-NoX-20B进行微调。同时，不同ChatGPT的强化学习，OpenChatKit采用一个60亿参数的审核模型，对不合适或者是有害的信息进行过滤，确保生成内容的安全和质量。

```
git clone git@github.com:togethercomputer/OpenChatKit.git
```





### BELLE

[BELLE](https://github.com/LianjiaTech/BELLE)基于 Stanford Alpaca ，实现基于Bloom、LLama的监督微调。Stanford Alpaca 的种子任务都是英语，收集的数据也都是英文，该开源项目是促进中文对话大模型开源社区的发展，针对中文做了优化，模型调优仅使用由ChatGPT生产的数据（不包含任何其他数据）。

```
git clone git@github.com:LianjiaTech/BELLE.git
```





### alpaca-lora和Chinese-alpaca-lora

[alpaca-lora](https://github.com/tloen/alpaca-lora), 基于lora低成本直接复现alpaca, 还有对中文进行优化和复现的版本。

```
git clone git@github.com:tloen/alpaca-lora.git
git clone git@github.com:LC1332/Chinese-alpaca-lora.git
```



### Dolly

Dolly在Alpaca的启发下，用Alpaca数据集，在GPT-J-6B上实现微调，由于Dolly本身是一个模型的“克隆”，所以团队最终决定将其命名为“多莉”。这种克隆式在Alpaca启发下越来越多，总结起来大致采用Alpaca开源的数据获取方式，在6B或者7B规模大小的旧模型上进行指令微调，获得类似ChatGPT的的效果。这种思想很经济，也能迅速模仿出ChatGPT的韵味来，广受欢迎，一经推出star爆棚。

```
git clone git@github.com:databrickslabs/dolly.git
```



### LMFLOW

[LMFLOW](https://github.com/OptimalScale/LMFlow)是香港科技大学开源的训练框架, 能够方便的微调语言大模型。

```
git clone git@github.com:OptimalScale/LMFlow.git
```





### Baize白泽

[白泽](https://github.com/project-baize/baize-chatbot)是UCSD, Sun Yat-sen University, MSRA发布的模型, 主打利用chatgpt和自己对话生成数据对下游的任务进行微调。

```
git clone git@github.com:project-baize/baize-chatbot.git
```



### Koala考拉

UC伯克利发布了一个可以在消费级GPU上运行的对话模型Koala，参数达到13B。Koala 的训练数据集包括如下几个部分：ChatGPT数据和开源数据（Open Instruction Generalist (OIG)、斯坦福 Alpaca 模型使用的数据集、Anthropic HH、OpenAI WebGPT、OpenAI Summarization）。Koala模型在EasyLM中使用JAX/Flax实现，用了8 个A100 GPU，完成2轮迭代需要6个小时。评测效果优于Alpaca，达到ChatGPT 50%的性能。

```
git clone git@github.com:young-geng/EasyLM.git
```



### StackLLaMA

随着斯坦福Alpaca的出现，一大堆基于LLama的羊驼家族和扩展动物家族开始出现，终于Hugging Face研究人员近期发布了一篇博客StackLLaMA：用RLHF训练LLaMA的实践指南。同时也发布了一个70亿参数的模型——StackLLaMA。这是一个通过人类反馈强化学习在LLaMA-7B微调而来的模型。

 https://huggingface.co/blog/stackllama



### Deep Speed Chat

deep speed对训练chat模型进行了优化, 声称能够达到15倍的加速, 极大节约成本，但这主要是一个训练框架, 并没有release模型。





### RRHF

RRHF提出了一种比PPO更加简单的和人类偏好对齐的算法, 并训练了自己的模型。

```
git clone git@github.com:GanjinZero/RRHF.git
```





### LLMZoo（凤凰Phoenix和Chimera）

[LLMZoo](https://github.com/FreedomIntelligence/LLMZoo)，即LLM动物园开源项目维护了一系列开源大模型，其中包括了近期备受关注的来自香港中文大学（深圳）和深圳市大数据研究院的王本友教授团队开发的Phoenix（凤凰）和Chimera等开源大语言模型，其中文本效果号称接近百度文心一言，GPT-4评测号称达到了97%文心一言的水平，在人工评测中五成不输文心一言。

```
git clone git@github.com:FreedomIntelligence/LLMZoo.git
```





### OpenAssistant

LAION开源的模型，OpenAssistant是一个开源聊天助手，其可以理解任务、与第三方系统交互、动态检索信息。据其说，其是第一个在人类数据上进行训练的完全开源的大规模指令微调模型。该模型主要创新在于一个较大的人类反馈数据集（详细说明见数据篇），公开测试显示效果在人类对齐和毒性方面做的不错，但是中文效果尚有不足。

```
git clone git@github.com:LAION-AI/Open-Assistant.git
```







### 参考

- https://github.com/chenking2020/FindTheChatGPTer
- 

