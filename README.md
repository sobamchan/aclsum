# ACLSum: A New Dataset for Aspect-based Summarization of Scientific Publications

This repository contains data for our paper "ACLSum: A New Dataset for Aspect-based Summarization of Scientific Publications" and a small 
utility class to work with it.


## Hugging Face datasets

You can also use Huggin Face datasets to load ACLSum ([dataset link](https://huggingface.co/datasets/sobamchan/aclsum)).
This would be convenient if you want to train transformer models using our dataset.

Just do,

```py
from datasets import load_dataset
dataset = load_dataset("sobamchan/aclsum")
```


## Our utility class

If you want to see what's in our data more carefully, the following example code on how to use our utility class may be helpful.

You can install the library with the dataset via pip, just run,

```bash
pip install aclsum
```

then you can load the dataset from your python code as,

```py
from aclsum import ACLSum

# Load per split ("train", "val", "test")
train = ACLSum("train")

# One data sample (= paper)
document = train[0]

# Three summaries on each aspect (dict[aspect, summary])
document.summaries

# Get all the sentences from the paper (we only work with abstract, introduction, and conclusion sections) (list[str])
document.get_all_sentences() 

# You can specify sections to extract sentences from
document.get_all_sentences(["abstract", "conclusion"])

# Get highlight labels (list[0 or 1])
document.get_all_highlights()

# Get highlighted sentences (list[str])
document.get_all_highlighted_sentences()
```
