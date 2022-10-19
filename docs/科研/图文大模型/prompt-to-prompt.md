# prompt-to-prompt



## 前言

这一部分主要是聚焦研究在diffusion model生成的过程中Attention map是如何起作用的, 可以用什么样的方法更好的利用attentionmap来控制模型的生成。这个部分最主要的启发工作来源于google的“Prompt-to-Prompt Image Editing with Cross Attention Control”, 也是我们希望复现和学习的文章。



## 复现和熟悉过程

之前在github上看到了有人对google的这篇文章进行了复现, 在这里进行尝试, 复现他的结果, 然后研究一下他是怎么做, 以及我们可以怎么做吧。直接上代码。

```shell
git clone git@github.com:bloc97/CrossAttentionControl.git
pip install torch transformers diffusers==0.4.1 numpy pillow tqdm jupyter
jupyter-notebook
# 如果是ssh在服务器上, 则需要通过ssh传递
jupyter-notebook --no-browser --port=1234 # on server
ssh -NL localhost:1234:localhost:1234 g5 # on your pc, then open link in server jupyter, notice port need to be your host port
```

跑是能跑的, 不断按shift + enter, 除了模型下载慢一点, 其他都可以。

### 粗读代码

引入包, load模型

```python
import torch
from transformers import CLIPModel, CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel
#NOTE: Last tested working diffusers version is diffusers==0.4.1, https://github.com/huggingface/diffusers/releases/tag/v0.4.1

#Init CLIP tokenizer and model
model_path_clip = "openai/clip-vit-large-patch14"
clip_tokenizer = CLIPTokenizer.from_pretrained(model_path_clip)
clip_model = CLIPModel.from_pretrained(model_path_clip, torch_dtype=torch.float16)
clip = clip_model.text_model

#Init diffusion model
auth_token = "这个是hugging face 的access token" #Replace this with huggingface auth token as a string if model is not already downloaded
model_path_diffusion = "CompVis/stable-diffusion-v1-4"
# 看起来stable diffusion用的diffusion model就是这一个, 但是对于模型参数而言, 不知道他到底下载的是参数的那一部分, 而且用了半精度
# 代码在这里 https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unet_2d_condition.py
unet = UNet2DConditionModel.from_pretrained(model_path_diffusion, subfolder="unet", use_auth_token=auth_token, revision="fp16", torch_dtype=torch.float16)
# 这看起来就是stable diffusion用的vae
vae = AutoencoderKL.from_pretrained(model_path_diffusion, subfolder="vae", use_auth_token=auth_token, revision="fp16", torch_dtype=torch.float16)

#Move to GPU
device = "cuda"
unet.to(device)
vae.to(device)
clip.to(device)
print("Loaded all models")
```



写的一些有关attention的函数

```python
import numpy as np
import random
from PIL import Image
from diffusers import LMSDiscreteScheduler # https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_lms_discrete.py
# 这个类的代码在这里, 但是具体的作用目前还不知道
from tqdm.auto import tqdm
from torch import autocast # 混合精度的东西
from difflib import SequenceMatcher # 比较两个序列

def init_attention_weights(weight_tuples):
  """
  初始化权重, 如果提供了对应的位置的权重, 则替换, 否则全为1
  """
    tokens_length = clip_tokenizer.model_max_length
    weights = torch.ones(tokens_length)
    
    for i, w in weight_tuples:
        if i < tokens_length and i >= 0:
            weights[i] = w
    
    # TODO 权重是一个序列长度的一维向量, 被初始化到last_attn_slice_weights, 这是什么东西
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention" and "attn2" in name:
            module.last_attn_slice_weights = weights.to(device)
        if module_name == "CrossAttention" and "attn1" in name:
            module.last_attn_slice_weights = None
    

def init_attention_edit(tokens, tokens_edit):
		"""
		
		"""
    tokens_length = clip_tokenizer.model_max_length
    mask = torch.zeros(tokens_length)
    indices_target = torch.arange(tokens_length, dtype=torch.long) # 0, 1, 2...
    indices = torch.zeros(tokens_length, dtype=torch.long) # 0, 0, 0 ...

    tokens = tokens.input_ids.numpy()[0] # 字典index?
    tokens_edit = tokens_edit.input_ids.numpy()[0]
    
    for name, a0, a1, b0, b1 in SequenceMatcher(None, tokens, tokens_edit).get_opcodes():
        if b0 < tokens_length:
            if name == "equal" or (name == "replace" and a1-a0 == b1-b0):
                mask[b0:b1] = 1
                indices[b0:b1] = indices_target[a0:a1]

    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention" and "attn2" in name:
            module.last_attn_slice_mask = mask.to(device)
            module.last_attn_slice_indices = indices.to(device)
        if module_name == "CrossAttention" and "attn1" in name:
            module.last_attn_slice_mask = None
            module.last_attn_slice_indices = None

# TODO 弄清last_attn_slice是什么, sliced_attention, attention两个函数的区别
def init_attention_func():
    #ORIGINAL SOURCE CODE: https://github.com/huggingface/diffusers/blob/91ddd2a25b848df0fa1262d4f1cd98c7ccb87750/src/diffusers/models/attention.py#L276
    def new_attention(self, query, key, value):
        # TODO: use baddbmm for better performance
        # query 和 key 通过矩阵乘, 再通过softmax得到attention map
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        attn_slice = attention_scores.softmax(dim=-1)
        # compute attention output
        
        if self.use_last_attn_slice:
            if self.last_attn_slice_mask is not None:
                new_attn_slice = torch.index_select(self.last_attn_slice, -1, self.last_attn_slice_indices)
                attn_slice = attn_slice * (1 - self.last_attn_slice_mask) + new_attn_slice * self.last_attn_slice_mask
            else:
                attn_slice = self.last_attn_slice

            self.use_last_attn_slice = False
				
        # 这一步是在进行edit之前的执行的一次使用原始prompt进行forward，然后保存attn map
        if self.save_last_attn_slice:
            self.last_attn_slice = attn_slice
            self.save_last_attn_slice = False

        if self.use_last_attn_weights and self.last_attn_slice_weights is not None:
            attn_slice = attn_slice * self.last_attn_slice_weights
            self.use_last_attn_weights = False
        
        # 如果没有injection, 则直接再矩阵乘value, 就得到了下一层的输出
        hidden_states = torch.matmul(attn_slice, value)
        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states
    
    def new_sliced_attention(self, query, key, value, sequence_length, dim):
        
        batch_size_attention = query.shape[0]
        hidden_states = torch.zeros(
            (batch_size_attention, sequence_length, dim // self.heads), device=query.device, dtype=query.dtype
        )
        slice_size = self._slice_size if self._slice_size is not None else hidden_states.shape[0]
        for i in range(hidden_states.shape[0] // slice_size):
            start_idx = i * slice_size
            end_idx = (i + 1) * slice_size
            attn_slice = (
                torch.matmul(query[start_idx:end_idx], key[start_idx:end_idx].transpose(1, 2)) * self.scale
            )  # TODO: use baddbmm for better performance
            attn_slice = attn_slice.softmax(dim=-1)
            
            if self.use_last_attn_slice:
                if self.last_attn_slice_mask is not None:
                    new_attn_slice = torch.index_select(self.last_attn_slice, -1, self.last_attn_slice_indices)
                    attn_slice = attn_slice * (1 - self.last_attn_slice_mask) + new_attn_slice * self.last_attn_slice_mask
                else:
                    attn_slice = self.last_attn_slice
                
                self.use_last_attn_slice = False
                    
            if self.save_last_attn_slice:
                self.last_attn_slice = attn_slice
                self.save_last_attn_slice = False
                
            if self.use_last_attn_weights and self.last_attn_slice_weights is not None:
                attn_slice = attn_slice * self.last_attn_slice_weights
                self.use_last_attn_weights = False
            
            attn_slice = torch.matmul(attn_slice, value[start_idx:end_idx])

            hidden_states[start_idx:end_idx] = attn_slice

        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states

    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention":
            module.last_attn_slice = None
            module.use_last_attn_slice = False
            module.use_last_attn_weights = False
            module.save_last_attn_slice = False
 # 以下是一个描述器功能, 此后调用module._sliced_attention, 等价于调用new_sliced_attention
 # 但是为什么不能直接把函数地址传进去呢, 也就是module._sliced_attention = new_sliced_attention
 # 这是因为函数中有参数self,如果按地址传, self也需要显示参数提供, 而使用描述器就可以直接传我们定义的参数了
            module._sliced_attention = new_sliced_attention.__get__(module, type(module))
            module._attention = new_attention.__get__(module, type(module))
            
def use_last_tokens_attention(use=True):
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention" and "attn2" in name:
            module.use_last_attn_slice = use
            
def use_last_tokens_attention_weights(use=True):
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention" and "attn2" in name:
            module.use_last_attn_weights = use
            
def use_last_self_attention(use=True):
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention" and "attn1" in name:
            module.use_last_attn_slice = use
            
def save_last_tokens_attention(save=True):
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention" and "attn2" in name:
            module.save_last_attn_slice = save
            
def save_last_self_attention(save=True):
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention" and "attn1" in name:
            module.save_last_attn_slice = save
```



这里我们需要看一下unet中的CrossAttention是如何定义的, 有哪些方法, 以及具体的运算逻辑时怎么样的。

```python
# https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unet_2d_condition.py
class CrossAttention(nn.Module):
    r"""
    A cross attention layer.
    Parameters:
        query_dim (:obj:`int`): The number of channels in the query.
        context_dim (:obj:`int`, *optional*):
            The number of channels in the context. If not given, defaults to `query_dim`.
        heads (:obj:`int`,  *optional*, defaults to 8): The number of heads to use for multi-head attention.
        dim_head (:obj:`int`,  *optional*, defaults to 64): The number of channels in each head.
        dropout (:obj:`float`, *optional*, defaults to 0.0): The dropout probability to use.
    """

    def __init__(
        self, query_dim: int, context_dim: Optional[int] = None, heads: int = 8, dim_head: int = 64, dropout: int = 0.0
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim if context_dim is not None else query_dim

        self.scale = dim_head**-0.5
        self.heads = heads
        # for slice_size > 0 the attention score computation
        # is split across the batch axis to save memory
        # You can set slice_size with `set_attention_slice`
        self._slice_size = None

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

    def reshape_heads_to_batch_dim(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size)
        return tensor

    def reshape_batch_dim_to_heads(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor

    def forward(self, hidden_states, context=None, mask=None):
        batch_size, sequence_length, _ = hidden_states.shape

        query = self.to_q(hidden_states)
        context = context if context is not None else hidden_states
        key = self.to_k(context)
        value = self.to_v(context)

        dim = query.shape[-1]

        query = self.reshape_heads_to_batch_dim(query)
        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        # TODO(PVP) - mask is currently never used. Remember to re-implement when used

        # attention, what we cannot get enough of

        if self._slice_size is None or query.shape[0] // self._slice_size == 1:
            hidden_states = self._attention(query, key, value)
        else:
            hidden_states = self._sliced_attention(query, key, value, sequence_length, dim)

        return self.to_out(hidden_states)
# 此函数被新写的函数替换
    def _attention(self, query, key, value):
        # TODO: use baddbmm for better performance
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        attention_probs = attention_scores.softmax(dim=-1)
        # compute attention output
        hidden_states = torch.matmul(attention_probs, value)
        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states
# 此函数被新写的函数替换
    def _sliced_attention(self, query, key, value, sequence_length, dim):
        batch_size_attention = query.shape[0]
        hidden_states = torch.zeros(
            (batch_size_attention, sequence_length, dim // self.heads), device=query.device, dtype=query.dtype
        )
        slice_size = self._slice_size if self._slice_size is not None else hidden_states.shape[0]
        for i in range(hidden_states.shape[0] // slice_size):
            start_idx = i * slice_size
            end_idx = (i + 1) * slice_size
            attn_slice = (
                torch.matmul(query[start_idx:end_idx], key[start_idx:end_idx].transpose(1, 2)) * self.scale
            )  # TODO: use baddbmm for better performance
            attn_slice = attn_slice.softmax(dim=-1)
            attn_slice = torch.matmul(attn_slice, value[start_idx:end_idx])

            hidden_states[start_idx:end_idx] = attn_slice

        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states
```



接下来是看整个stablediffusion的生成过程是如何运作的。

```python
@torch.no_grad()
def stablediffusion(prompt="", prompt_edit=None, prompt_edit_token_weights=[], prompt_edit_tokens_start=0.0, prompt_edit_tokens_end=1.0, prompt_edit_spatial_start=0.0, prompt_edit_spatial_end=1.0, guidance_scale=7.5, steps=50, seed=None, width=512, height=512, init_image=None, init_image_strength=0.5):
    #Change size to multiple of 64 to prevent size mismatches inside model
    width = width - width % 64
    height = height - height % 64
    
    #If seed is None, randomly select seed from 0 to 2^32-1
    if seed is None: seed = random.randrange(2**32 - 1)
    generator = torch.cuda.manual_seed(seed)
    
    #Set inference timesteps to scheduler
    scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
    scheduler.set_timesteps(steps)
    
    #Preprocess image if it exists (img2img)
    if init_image is not None:
        #Resize and transpose for numpy b h w c -> torch b c h w
        init_image = init_image.resize((width, height), resample=Image.Resampling.LANCZOS)
        init_image = np.array(init_image).astype(np.float32) / 255.0 * 2.0 - 1.0
        init_image = torch.from_numpy(init_image[np.newaxis, ...].transpose(0, 3, 1, 2))
        
        #If there is alpha channel, composite alpha for white, as the diffusion model does not support alpha channel
        if init_image.shape[1] > 3:
            init_image = init_image[:, :3] * init_image[:, 3:] + (1 - init_image[:, 3:])
            
        #Move image to GPU
        init_image = init_image.to(device)
        
        #Encode image
        with autocast(device):
            init_latent = vae.encode(init_image).latent_dist.sample(generator=generator) * 0.18215
            
        t_start = steps - int(steps * init_image_strength)
            
    else:
        init_latent = torch.zeros((1, unet.in_channels, height // 8, width // 8), device=device)
        t_start = 0
    
    #Generate random normal noise
    noise = torch.randn(init_latent.shape, generator=generator, device=device)
    #latent = noise * scheduler.init_noise_sigma
    latent = scheduler.add_noise(init_latent, noise, torch.tensor([scheduler.timesteps[t_start]], device=device)).to(device)
    
    #Process clip
    with autocast(device):
        tokens_unconditional = clip_tokenizer("", padding="max_length", max_length=clip_tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True)
        embedding_unconditional = clip(tokens_unconditional.input_ids.to(device)).last_hidden_state

        tokens_conditional = clip_tokenizer(prompt, padding="max_length", max_length=clip_tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True)
        embedding_conditional = clip(tokens_conditional.input_ids.to(device)).last_hidden_state

        #Process prompt editing
        if prompt_edit is not None:
            tokens_conditional_edit = clip_tokenizer(prompt_edit, padding="max_length", max_length=clip_tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True)
            embedding_conditional_edit = clip(tokens_conditional_edit.input_ids.to(device)).last_hidden_state
            
            init_attention_edit(tokens_conditional, tokens_conditional_edit)
            
        init_attention_func()
        init_attention_weights(prompt_edit_token_weights)
            
        timesteps = scheduler.timesteps[t_start:]
        
        for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):
            t_index = t_start + i

            #sigma = scheduler.sigmas[t_index]
            latent_model_input = latent
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            #Predict the unconditional noise residual
            # 这里是为了使用classifier free guidance
            noise_pred_uncond = unet(latent_model_input, t, encoder_hidden_states=embedding_unconditional).sample
            
            #Prepare the Cross-Attention layers
            if prompt_edit is not None:
                save_last_tokens_attention()
                save_last_self_attention()
            else:
                #Use weights on non-edited prompt when edit is None
                use_last_tokens_attention_weights()
                
            #Predict the conditional noise residual and save the cross-attention layer activations
            noise_pred_cond = unet(latent_model_input, t, encoder_hidden_states=embedding_conditional).sample
            
            #Edit the Cross-Attention layer activations
            if prompt_edit is not None:
                t_scale = t / scheduler.num_train_timesteps
                if t_scale >= prompt_edit_tokens_start and t_scale <= prompt_edit_tokens_end:
                    use_last_tokens_attention()
                if t_scale >= prompt_edit_spatial_start and t_scale <= prompt_edit_spatial_end:
                    use_last_self_attention()
                    
                #Use weights on edited prompt
                use_last_tokens_attention_weights()

                #Predict the edited conditional noise residual using the cross-attention masks
                noise_pred_cond = unet(latent_model_input, t, encoder_hidden_states=embedding_conditional_edit).sample
                
            #Perform guidance
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            latent = scheduler.step(noise_pred, t_index, latent).prev_sample

        #scale and decode the image latents with vae
        # TODO 0.18215 这个数字是怎么来的呢
        latent = latent / 0.18215
        image = vae.decode(latent.to(vae.dtype)).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image[0] * 255).round().astype("uint8")
    return Image.fromarray(image)

```







## 需要关注的问题

- 弄清last_attn_slice是什么, sliced_attention, attention两个函数的区别
- 弄清楚VAE的原理, 弄清楚attention map中的位置是否体现出原图中的空域信息
- 弄清楚attention有几层,  每个层的attention层的变化情况, 以及哪些层是重要的





话不多说, 直接写代码来解决问题。

```python
for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention":
            if "attn2" in name:
                attn2_layers += 1
            attn_layers += 1
            module.last_attn_slice = None
            module.use_last_attn_slice = False
            module.use_last_attn_weights = False
            module.save_last_attn_slice = False
            module._sliced_attention = new_sliced_attention.__get__(module, type(module))
            module._attention = new_attention.__get__(module, type(module))
    print("attn layers:", attn_layers)
    print("atten2 layers:", attn2_layers)
```

输出:

```
attn layers: 32
atten2 layers: 16
根据后续的输出, 可知attn1和attn2是交替出现的
```

同时查看query, key, value的维度参数, 我们可以注意到这是`selfAttention` 以及 `crossAttention`, 交替使用, 这也印证了代码中需要区分attn1以及attn2, 这也是为什么主要inject的是attn2的layer, 而不是attn1。

```
attn1  query.shape torch.Size([8, 4096, 40]) 
key.shape torch.Size([8, 4096, 40]) 
value.shape torch.Size([8, 4096, 40])
1 origin attn map: torch.Size([8, 4096, 4096])
new attn map: torch.Size([8, 4096, 4096])

attn2  query.shape torch.Size([8, 4096, 40]) 
key.shape torch.Size([8, 77, 40]) 
value.shape torch.Size([8, 77, 40])
2 origin attn map: torch.Size([8, 4096, 77])
new attn map: torch.Size([8, 4096, 77])

attn1  query.shape torch.Size([8, 4096, 40]) 
key.shape torch.Size([8, 4096, 40]) 
value.shape torch.Size([8, 4096, 40])
3 origin attn map: torch.Size([8, 4096, 4096])
new attn map: torch.Size([8, 4096, 4096])

attn2  query.shape torch.Size([8, 4096, 40]) 
key.shape torch.Size([8, 77, 40]) 
value.shape torch.Size([8, 77, 40])
4 origin attn map: torch.Size([8, 4096, 77])
new attn map: torch.Size([8, 4096, 77])

attn1  query.shape torch.Size([8, 1024, 80]) 
key.shape torch.Size([8, 1024, 80]) 
value.shape torch.Size([8, 1024, 80])
5 origin attn map: torch.Size([8, 1024, 1024])
new attn map: torch.Size([8, 1024, 1024])

attn2  query.shape torch.Size([8, 1024, 80]) 
key.shape torch.Size([8, 77, 80]) 
value.shape torch.Size([8, 77, 80])
6 origin attn map: torch.Size([8, 1024, 77])
new attn map: torch.Size([8, 1024, 77])

attn1  query.shape torch.Size([8, 1024, 80]) 
key.shape torch.Size([8, 1024, 80]) 
value.shape torch.Size([8, 1024, 80])
7 origin attn map: torch.Size([8, 1024, 1024])
new attn map: torch.Size([8, 1024, 1024])

attn2  query.shape torch.Size([8, 1024, 80]) 
key.shape torch.Size([8, 77, 80]) 
value.shape torch.Size([8, 77, 80])
8 origin attn map: torch.Size([8, 1024, 77])
new attn map: torch.Size([8, 1024, 77])

attn1  query.shape torch.Size([8, 256, 160]) 
key.shape torch.Size([8, 256, 160]) 
value.shape torch.Size([8, 256, 160])
9 origin attn map: torch.Size([8, 256, 256])
new attn map: torch.Size([8, 256, 256])

attn2  query.shape torch.Size([8, 256, 160]) 
key.shape torch.Size([8, 77, 160]) 
value.shape torch.Size([8, 77, 160])
10 origin attn map: torch.Size([8, 256, 77])
new attn map: torch.Size([8, 256, 77])

attn1  query.shape torch.Size([8, 256, 160]) 
key.shape torch.Size([8, 256, 160]) 
value.shape torch.Size([8, 256, 160])
11 origin attn map: torch.Size([8, 256, 256])
new attn map: torch.Size([8, 256, 256])

attn2  query.shape torch.Size([8, 256, 160]) 
key.shape torch.Size([8, 77, 160]) 
value.shape torch.Size([8, 77, 160])
12 origin attn map: torch.Size([8, 256, 77])
new attn map: torch.Size([8, 256, 77])

attn1  query.shape torch.Size([8, 64, 160]) 
key.shape torch.Size([8, 64, 160]) 
value.shape torch.Size([8, 64, 160])
31 origin attn map: torch.Size([8, 64, 64])
new attn map: torch.Size([8, 64, 64])

attn2  query.shape torch.Size([8, 64, 160]) 
key.shape torch.Size([8, 77, 160]) 
value.shape torch.Size([8, 77, 160])
32 origin attn map: torch.Size([8, 64, 77])
new attn map: torch.Size([8, 64, 77])

attn1  query.shape torch.Size([8, 256, 160]) 
key.shape torch.Size([8, 256, 160]) 
value.shape torch.Size([8, 256, 160])
13 origin attn map: torch.Size([8, 256, 256])
new attn map: torch.Size([8, 256, 256])

attn2  query.shape torch.Size([8, 256, 160]) 
key.shape torch.Size([8, 77, 160]) 
value.shape torch.Size([8, 77, 160])
14 origin attn map: torch.Size([8, 256, 77])
new attn map: torch.Size([8, 256, 77])

attn1  query.shape torch.Size([8, 256, 160]) 
key.shape torch.Size([8, 256, 160]) 
value.shape torch.Size([8, 256, 160])
15 origin attn map: torch.Size([8, 256, 256])
new attn map: torch.Size([8, 256, 256])

attn2  query.shape torch.Size([8, 256, 160]) 
key.shape torch.Size([8, 77, 160]) 
value.shape torch.Size([8, 77, 160])
16 origin attn map: torch.Size([8, 256, 77])
new attn map: torch.Size([8, 256, 77])

attn1  query.shape torch.Size([8, 256, 160]) 
key.shape torch.Size([8, 256, 160]) 
value.shape torch.Size([8, 256, 160])
17 origin attn map: torch.Size([8, 256, 256])
new attn map: torch.Size([8, 256, 256])

attn2  query.shape torch.Size([8, 256, 160]) 
key.shape torch.Size([8, 77, 160]) 
value.shape torch.Size([8, 77, 160])
18 origin attn map: torch.Size([8, 256, 77])
new attn map: torch.Size([8, 256, 77])

attn1  query.shape torch.Size([8, 1024, 80]) 
key.shape torch.Size([8, 1024, 80]) 
value.shape torch.Size([8, 1024, 80])
19 origin attn map: torch.Size([8, 1024, 1024])
new attn map: torch.Size([8, 1024, 1024])

attn2  query.shape torch.Size([8, 1024, 80]) 
key.shape torch.Size([8, 77, 80]) 
value.shape torch.Size([8, 77, 80])
20 origin attn map: torch.Size([8, 1024, 77])
new attn map: torch.Size([8, 1024, 77])

attn1  query.shape torch.Size([8, 1024, 80]) 
key.shape torch.Size([8, 1024, 80]) 
value.shape torch.Size([8, 1024, 80])
21 origin attn map: torch.Size([8, 1024, 1024])
new attn map: torch.Size([8, 1024, 1024])

attn2  query.shape torch.Size([8, 1024, 80]) 
key.shape torch.Size([8, 77, 80]) 
value.shape torch.Size([8, 77, 80])
22 origin attn map: torch.Size([8, 1024, 77])
new attn map: torch.Size([8, 1024, 77])

attn1  query.shape torch.Size([8, 1024, 80]) 
key.shape torch.Size([8, 1024, 80]) 
value.shape torch.Size([8, 1024, 80])
23 origin attn map: torch.Size([8, 1024, 1024])
new attn map: torch.Size([8, 1024, 1024])

attn2  query.shape torch.Size([8, 1024, 80]) 
key.shape torch.Size([8, 77, 80]) 
value.shape torch.Size([8, 77, 80])
24 origin attn map: torch.Size([8, 1024, 77])
new attn map: torch.Size([8, 1024, 77])

attn1  query.shape torch.Size([8, 4096, 40]) 
key.shape torch.Size([8, 4096, 40]) 
value.shape torch.Size([8, 4096, 40])
25 origin attn map: torch.Size([8, 4096, 4096])
new attn map: torch.Size([8, 4096, 4096])

attn2  query.shape torch.Size([8, 4096, 40]) 
key.shape torch.Size([8, 77, 40]) 
value.shape torch.Size([8, 77, 40])
26 origin attn map: torch.Size([8, 4096, 77])
new attn map: torch.Size([8, 4096, 77])
2
attn1  query.shape torch.Size([8, 4096, 40]) 
key.shape torch.Size([8, 4096, 40]) 
value.shape torch.Size([8, 4096, 40])
27 origin attn map: torch.Size([8, 4096, 4096])
new attn map: torch.Size([8, 4096, 4096])

attn2  query.shape torch.Size([8, 4096, 40]) 
key.shape torch.Size([8, 77, 40]) 
value.shape torch.Size([8, 77, 40])
28 origin attn map: torch.Size([8, 4096, 77])
new attn map: torch.Size([8, 4096, 77])

attn1  query.shape torch.Size([8, 4096, 40]) 
key.shape torch.Size([8, 4096, 40]) 
value.shape torch.Size([8, 4096, 40])
29 origin attn map: torch.Size([8, 4096, 4096])
new attn map: torch.Size([8, 4096, 4096])

attn2  query.shape torch.Size([8, 4096, 40]) 
key.shape torch.Size([8, 77, 40]) 
value.shape torch.Size([8, 77, 40])
30 origin attn map: torch.Size([8, 4096, 77])
new attn map: torch.Size([8, 4096, 77])
```







## 注意到的问题

### 其实生成控制没有很理想

我们注意到代码中给的样例看起来不错, 但是稍微添加一些修改, 就会出现一些问题, 当然这任然需要再回头看一下论文并进行修改。

例如, 原来的实现中, 使用的原条件生成`“a cat sitting on a car”,    seed=248396402679`, 得到如下图片:

![image-20221019113709789](./prompt-to-prompt.assets/image-20221019113709789.png)

attention inject的方式, 参数为`"a cat sitting on a car", "a smiling dog sitting on a car", prompt_edit_spatial_start=0.7, seed=248396402679` 则如下图:

![image-20221019113909102](./prompt-to-prompt.assets/image-20221019113909102.png)

如果将参数改变为`"a cat sitting on a car", "a dog sitting on a car", prompt_edit_spatial_start=0.7,seed=248396402679,steps=50`, 则得到如下图片, 比较恐怖:

![image-20221019114052787](./prompt-to-prompt.assets/image-20221019114052787.png)



value, map不同组合, 抛弃selfattention , new_sliced_attention , seq compare
