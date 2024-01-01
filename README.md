# LORA-turkish-clip

# What is this project? 

In this repo we finetune OpenAi's CLIP model for Turkish language using LORA method. To get more information on CLIP you can checkout my other [repo]. 

Flicker8 dataset is used for training. For Turkish captions TasvirEt dataset is used which is a dataset for Turkish Flicker8 captions.

The notebooks are inteded to run on Colab although it is not necessary. Necessary library installations are done in notebooks in order to run on Colab.


# What is LORA?

Low-Rank Adaptation is a technique to finetune models. Instead of finetuning all layers some layers are selected and finetuned. Compared to regular finetuning instead of changing the original weights the difference between the initial weights and finetuned weights are stored and saved. Also matrices storing the weights are decomposed into two matrices which reduce the size of the model weights and ensures that the initial and finetuned model to be similar. This means that the model will be much more protective against cathastropical forgetting. All of these also reduce the over all model size. So you can just save the difference weight (or LORA weight in other words) instead of saving whole finetuned version's weights. This gives the ability to have a single large base model and multiple small LORA weights instead of saving large models for each finetuned version. 


# What is in the notebooks?

LORA_CLIP_training_Tasviret -> Finetuning code of CLIP with LORA


# How to use the model?

I have uploaded the model to [HuggingFace]. The model can be used like the example below:

```Python
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
model.load_adapter("kesimeg/lora-turkish-clip") # model can be found on Huggingface 🎉
model.eval()

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


img = Image.open("dog.png") # A dog image
inputs = processor(text=["Çimenler içinde bir köpek.","Bir köpek.","Çimenler içinde bir kuş."], images=img, return_tensors="pt", padding=True)
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1)
print(probs)


```

# What can be done more?
This goal of this repo is to show how to use LORA to finetune CLIP rather than getting a good CLIP model for Turkish language. You can try increasing performance by adding augmentations and measure performance with better metrics. You can also try replacing textual encoder with a model pretrained with Turkish data such as DistilBERT. (You can check my other [repo] for this) 

# Resources

I want to thank to sources below which I have used to make this project:

Finetuning an image classifier with LORA using PEFT library:
https://github.com/huggingface/peft/tree/main/examples/image_classification

PEFT library tutorial:
https://huggingface.co/docs/transformers/main/en/peft

Original LORA paper:
https://arxiv.org/abs/2106.09685

TasvirEt (Turkish Flicker8 captions):
https://www.kaggle.com/datasets/begum302553/tasviret-flickr8k-turkish 

TasvirEt paper:
https://ieeexplore.ieee.org/document/7496155

Original CLIP paper:
https://arxiv.org/abs/2103.00020


[repo]: https://github.com/kesimeg/turkish-clip
[HuggingFace]: https://huggingface.co/kesimeg/lora-turkish-clip
