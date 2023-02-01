# mT5-small based Azerbaijani Summarization

In this model, [Google's Multilingual T5-small](https://github.com/google-research/multilingual-t5) is fine-tuned on [Azerbaijani News Summary Dataset](https://huggingface.co/datasets/nijatzeynalov/azerbaijani-multi-news) for **Summarization** downstream task. The model is trained with 3 epochs, 64 batch size and 10e-4 learning rate. It took almost 12 hours on GPU instance with Ubuntu Server 20.04 LTS image in Microsoft Azure. The max news length is kept as 2048 and max summary length is determined as 128.


mT5 is a multilingual variant of __T5__ and only pre-trained on [mC4](https://www.tensorflow.org/datasets/catalog/c4#c4multilingual)
excluding any supervised training. Therefore, the mT5 model has to be fine-tuned before it is useable on a downstream task.

### Text-to-Text Transfer Transformer (T5)

The paper [“Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer”](https://arxiv.org/pdf/1910.10683.pdf) presents a large-scale empirical survey to determine which transfer learning techniques work best and apply these insights at scale to create a new model called the Text-To-Text Transfer Transformer.

![Alt Text](https://miro.medium.com/max/1280/0*xfXDPjASztwmJlOa.gif)



T5, or Text-to-Text Transfer Transformer, is a Transformer based architecture that uses a text-to-text approach. Every task – including translation, question answering, and classification – is cast as feeding the model text as input and training it to generate some target text. This allows for the use of the same model, loss function, hyperparameters, etc. across our diverse set of tasks. 

The changes compared to BERT include:

- adding a causal decoder to the bidirectional architecture.
- replacing the fill-in-the-blank cloze task with a mix of alternative pre-training tasks.

The model was trained on a cleaned version of Common Crawl that is two orders of magnitude larger than Wikipedia. 

The T5 model, pre-trained on C4, achieves state-of-the-art results on many NLP benchmarks while being flexible enough to be fine-tuned to several downstream tasks. The pre-trained T5 in Hugging Face is also trained on the mixture of unsupervised training (which is trained by reconstructing the masked sentence) and task-specific training.

### Multilingual t5

["mt5"](https://arxiv.org/pdf/2010.11934v3.pdf) is a multilingual variant of T5 that was pre-trained on a new Common Crawl-based dataset covering 
101 languages. 

mT5 is pre-trained only by unsupervised manner with multiple languages, and it’s not trained for specific downstream tasks. To dare say, this pre-trained model has ability to build correct text in Azerbaijani, but it doesn’t have any ability for specific tasks, such as, summarization, correction, machine translation, etc.

In HuggingFace, several sizes of mT5 models are available, and here I used small one (google/mt5-small). Therefore I trained (fine-tune) this model for summarization in Azerbaijani using [Azerbaijani News Summary Dataset](https://huggingface.co/datasets/nijatzeynalov/azerbaijani-multi-news).


## Training hyperparameters

__mT5-based-azerbaijani-summarize__ model training took almost 12 hours on GPU instance with Ubuntu Server 20.04 LTS image in Microsoft Azure. The following hyperparameters were used during training:

- learning_rate: 0.0005
- train_batch_size: 2
- eval_batch_size: 1
- seed: 42
- gradient_accumulation_steps: 16
- total_train_batch_size: 64
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 90
- num_epochs: 10

## Dataset

Model was trained on [__az-news-summary__ dataset](https://huggingface.co/datasets/nijatzeynalov/azerbaijani-multi-news), a comprehensive and diverse dataset comprising 143k (143,448) Azerbaijani news articles extracted using a set of carefully designed heuristics. 
 
The dataset covers common topics for news reports include war, government, politics, education, health, the environment, economy, business, fashion, entertainment, and sport, as well as quirky or unusual events.

This dataset has 3 splits: _train_, _validation_, and _test_. \
Token counts are white space based.

| Dataset Split | Number of Instances |     Size (MB)         |
| ------------- | --------------------|:----------------------|
| Train         | 100,413             |      150              |
| Validation    | 14,344              |      21.3             |
| Test          | 28,691              |      42.8             |


## Training results with comparison

__mT5-based-azerbaijani-summarize__ model rouge scores on the test set:

- Rouge1: 39.4222
- Rouge2: 24.8624
- Rougel: 32.2487

For __Azerbaijani text summarization downstream task__, mT5-multilingual-XLSum has also been developed on the 45 languages of [XL-Sum](https://huggingface.co/datasets/csebuetnlp/xlsum) dataset. For finetuning details and scripts,
see the [paper](https://aclanthology.org/2021.findings-acl.413/) and the [official repository](https://github.com/csebuetnlp/xl-sum). .

__mT5_multilingual_XLSum__ modelrouge scores on the XL-Sum test set (only for Azerbaijani):

- Rouge1: 21.4227
- Rouge2: 9.5214
- Rougel: 19.3331

As seen from the numbers, our model __mT5-based-azerbaijani-summarize__  achieves dramatically better performance than __mT5_multilingual_XLSum__.

## Using this model in transformers

```python
!pip install sentencepiece
!pip install transformers
```

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

article_text = """Ötən il Azərbaycana 74 577 avtomobil idxal edilib. Bu da 2021-ci illə müqayisədə 16 617 ədəd və ya 18,2% azdır.
Xezerxeber.az-ın məlumatına görə, avtomobil bazarı üzrə qiymətləndirici Sərxan Qədirov deyib ki, əvvəl ay ərzində 5-10 avtomobil gətirən şəxslər hazırda bu sayı 2-3 ədədə endiriblər. Hətta ölkəyə nəqliyyat vasitələrinin gətirilməsi işini dayandıranlar da var.
Nəqliyyat məsələləri üzrə ekspert Eldəniz Cəfərov isə bildirib ki, gözləniləndən fərqli olaraq, ölkəyə idxal olunan kiçik mühərrikli avtomobillərin sayında da azalma var. Bunun başlıca səbəbi Rusiyada istehsalın dayandırılmasıdır.
Ekspertin sözlərinə görə, əvvəllər Azərbaycan bazarında Rusiya istehsalı olan nəqliyyat vasitələri geniş yer tuturdu. Hazırda isə həmin ölkədən idxal tam dayanıb."""

model_name = "nijatzeynalov/mT5-based-azerbaijani-summarize"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
```

```python
input_ids = tokenizer(
    article_text,
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=2048
)["input_ids"]

output_ids = model.generate(
    input_ids=input_ids,
    max_length=128,
    no_repeat_ngram_size=2,
    num_beams=4
)[0]

summary = tokenizer.decode(
    output_ids,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)

print(summary)
```

Result:

```python
Azərbaycana idxal olunan avtomobillərin sayı açıqlanıb
```

## Citation

If you use this model, please cite:

```
@misc {nijatzeynalov_2023,
	author       = { {NijatZeynalov} },
	title        = { mT5-based-azerbaijani-summarize (Revision 19930ab) },
	year         = 2023,
	url          = { https://huggingface.co/nijatzeynalov/mT5-based-azerbaijani-summarize },
	doi          = { 10.57967/hf/0316 },
	publisher    = { Hugging Face }
}
```
