# BERTScore model

Download `xlnet-large-cased` into this directory:

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download xlnet/xlnet-large-cased \
    --local-dir BERTScore_model/xlnet-large-cased
```

Expected layout:

```
BERTScore_model/
└── xlnet-large-cased/
    ├── config.json
    ├── pytorch_model.bin
    ├── spiece.model
    ├── tokenizer.json
    └── xlnet-large-cased.tsv
```

Official source: <https://huggingface.co/xlnet/xlnet-large-cased>

`xlnet-large-cased.tsv` is the rescaling baseline used by `scripts/run_SemDesc_task_evaluation.sh`; without it, drop the `--bertscore-rescale-with-baseline` flag to report raw BERTScore.

The other two backbones in `src/task_evaluation/evaluation_metrics.py` (`scibert`, `deberta_xlarge_mnli`) are fetched from the Hugging Face Hub by `bert-score` automatically.
