# diffusion库



## 前言

最近diffusion model的火热让朱老师觉得应该在diffusion领域实现自己的库, 然后打算组织同学们来开发这个库, 首先这个库的定位和目标是讨论的重点, 第一次开会主要讨论了相关的内容。



### 第一次开会

主要分析了已有的库的优点和缺点。

目前主要的diffusion库有diffusers, 还有k-diffusion, 其中hugging face的diffusers是影响力比较大的, 同时还有对应的demo, pipline, 使得使用和学习都比较方便, 代码和hugging face的服务同步。

k-diffusion主要是一个人维护的, stable diffusion的官方都主要用的是这个库。

路橙学长主要担忧的是人手不够,  难以长期维护一个开源的社区, 朱老师表示可以调配工程师和本科生。对于定位和社区的问题, 朱老师表示可以参考tianshou的成功, tianshou当时主打的就是模块化和可复用, 路橙学长表示再加上文档和测试的工作才使得tianshou比较成功。

最后我们定位是, 给初级的科学研究者做一个使用方便, 容易学, 并且模型内部结构也容易复用的库, 同时还要做可复现以及evaluation。

