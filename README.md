# UMT

## Using the NMT models

If running in compute-canada clusters, download the models below from the launch machine.

```
config = AutoConfig.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-zh")

tokenizer.save_pretrained("/home/pparth2/scratch/UMT/UMT/Results/cached/Helsinki-NLP-opus-en-zh")
config.save_pretrained("/home/pparth2/scratch/UMT/UMT/Results/cached/Helsinki-NLP-opus-en-zh")
model.save_pretrained("/home/pparth2/scratch/UMT/UMT/Results/cached/Helsinki-NLP-opus-en-zh-model")
```

## Create the files:

`python datasets_preprocess.py`

For sanity check, open `Data/<model>/<lang>/stats.txt` and that should show the fraction of samples used by the different perturbations. If none of them are zero, then the create files works well.


## Run the translate:

Set the model and lang to generate the translations. If the GPU memory is < 32G, opt for a lower batch-size.

`python translate.py` should get the results.

## Compute Metrics:

Set the model and lang to generate the translations. If the GPU memory is < 32G, opt for a lower batch-size.

`python compute_metrics.py` should get the results in `Metrics\` in root.
