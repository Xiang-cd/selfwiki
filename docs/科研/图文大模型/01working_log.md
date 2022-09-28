# 工作日志



## 2022年09月21日

主要就是做了一些[textrual inverse](https://textual-inversion.github.io/)的复现, 其实就是把代码clone下来跑了一下, 这个文章主要解决的问题是在当前的diffusion model生成的图片多样性的是足够的, 但是如果我们想一直keep一个我们想要的物体在图片中, 需要怎么做呢。这篇文章提出了在text embedding空间中找到一个特定的embedding 来作为这个特定物体的符号代表, 寻找的方法是通过优化的方法, 可以认为用梯度优化embedding, 模型固定, loss 和 train diffusion 过程一致。

当前在项目中自己的定位还没有特别明晰, 自己对这个领域的idea也没有什么, 动手能力有待提高, 还是希望在push自己的过程中实现进步。
