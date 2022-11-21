# Textual inversion

- [toc]

## 前言

这一部分主要聚焦如何控制一个固定物体在生成过程中出现, 但是却不改变模型的参数, 这里的复现工作主要基于“An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion”, 代码则基于https://github.com/rinongal/textual_inversion, 使用的模型是ldm, 不是stable-diffusion。



我们在这里期望做到以下几点:

1. 复现代码, 看看效果
2. 尝试理解原理
3. 总结优点和缺点



因为文章思路很简单, 感觉没有十分值得讨论的地方, 我们只需要看以下效果。

![image-20221101182905847](./textual_inversion.assets/image-20221101182905847.png)





### cat

| ![1](./textual_inversion.assets/1.jpeg) | ![2](./textual_inversion.assets/2.jpeg) | ![3](./textual_inversion.assets/3.jpeg) | ![4](./textual_inversion.assets/4.jpeg) |
| --------------------------------------- | --------------------------------------- | --------------------------------------- | --------------------------------------- |
| ![5](./textual_inversion.assets/5.jpeg) | ![6](./textual_inversion.assets/6.jpeg) | ![7](./textual_inversion.assets/7.jpeg) |                                         |
|                                         |                                         |                                         |                                         |



#### a cake in style of cat

![a-cake-in-style-of-*](./textual_inversion.assets/a-cake-in-style-of-*.jpg)





### airpods

| ![IMG_1950](./textual_inversion.assets/IMG_1950.jpg) | ![IMG_1951](./textual_inversion.assets/IMG_1951.jpg) | ![IMG_1952](./textual_inversion.assets/IMG_1952.jpg) | ![IMG_1953](./textual_inversion.assets/IMG_1953.jpg) |
| ---------------------------------------------------- | ---------------------------------------------------- | ---------------------------------------------------- | ---------------------------------------------------- |
| ![IMG_1954](./textual_inversion.assets/IMG_1954.jpg) | ![IMG_1955](./textual_inversion.assets/IMG_1955.jpg) | ![IMG_1956](./textual_inversion.assets/IMG_1956.jpg) |                                                      |



#### a red airpods

![a-red-airpods](./textual_inversion.assets/a-red-airpods.jpg)

#### a picture of airpods

![a-picture-of-airpots](./textual_inversion.assets/a-picture-of-airpots.jpg)



### magic cube

| ![IMG_1960](./textual_inversion.assets/IMG_1960.jpg) | ![IMG_1961](./textual_inversion.assets/IMG_1961.jpg) | ![IMG_1962](./textual_inversion.assets/IMG_1962.jpg) |
| ---------------------------------------------------- | ---------------------------------------------------- | ---------------------------------------------------- |
| ![IMG_1963](./textual_inversion.assets/IMG_1963.jpg) | ![IMG_1964](./textual_inversion.assets/IMG_1964.jpg) | ![IMG_1965](./textual_inversion.assets/IMG_1965.jpg) |





#### photo of maigc cube

![photo-of-magic-cube](./textual_inversion.assets/photo-of-magic-cube.jpg)



#### red magic cube

![a-red-magic-cube](./textual_inversion.assets/a-red-magic-cube.jpg)



#### magic cube by origin

![a-photo-of-magic-cube](./textual_inversion.assets/a-photo-of-magic-cube.jpg)







### elephant

| ![1](./textual_inversion.assets/1.jpg)         | ![2](./textual_inversion.assets/2.jpg) | ![3](./textual_inversion.assets/3.jpg) |
| ---------------------------------------------- | -------------------------------------- | -------------------------------------- |
| ![4](./textual_inversion.assets/4-7298202.jpg) | ![5](./textual_inversion.assets/5.jpg) |                                        |



#### elephant drinking water

![a-elephant-is-drinking-water](./textual_inversion.assets/a-elephant-is-drinking-water.jpg)



#### elephant running

![a-elephant-is-running](./textual_inversion.assets/a-elephant-is-running.jpg)





## 结论

1. 不变参数, flexible
2. 有一定效果
3. 训练1-2小时
4. 有不理想的地方