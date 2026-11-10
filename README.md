# MatPhaseBench: A Semantics-Guided Benchmark for Materials Phase Diagrams Understanding

# Abstract

Materials phase diagrams encode complex scientific information about phase stability, transformations, and thermodynamic relationships, requiring both visual understanding and domain-specific reasoning. However, existing vision-language model (VLM) benchmarks provide limited evaluation of such complex scientific images. To address this gap, we introduce MatPhaseBench, a benchmark for open-ended materials phase diagram understanding. Starting from a large collection of scientific papers, we constructed thousands of phase-diagram--text samples using a two-stage image--text matching strategy that incorporates both direct figure references and contextually related discussions. We established MatPhaseBench-200 through expert evaluation and subsequently selected 800 additional samples using an expert-calibrated language-model judge to form MatPhaseBench-1000. To evaluate scientific understanding beyond conventional text similarity, we introduce a nine-dimensional Semantic Quality Score (SQS) based on VLM-as-Judge, assessing Semantic Completeness, Information Consistency, and Visual Content Alignment. Experiments on 13 representative VLMs reveal limitations in phase diagram understanding. On MatPhaseBench-1000, the highest BERTScore-R is 0.371, while the highest SQS reaches 12.996 out of 18 (72.2%). Further experiments show that semantic-dimension guidance improves performance, whereas additional few-shot demonstrations do not consistently provide benefits. These results reveal a gap between visual-content alignment and the coverage of scientific reasoning in model-generated phase-diagram descriptions.

## Repository layout

```
MatPhaseBench/
├── dataset/MatPhaseBench-SemDesc/
│   ├── task_dataset/
│   │   ├── phase_evaluation_data_200.json
│   │   └── phase_evaluation_expanded_data_800.json
│   ├── phase_images/                    # phase_0001.jpg … phase_1000.jpg
│   └── fewshot_images/                  # 3 few-shot examples
├── prompts/
│   ├── SemDesc_task_prompt.yaml         # description prompting
│   └── semantic_quality_score.yaml      # LLM-as-Judge rubric
├── scripts/
│   ├── run_SemDesc_task.sh              # inference
│   └── run_SemDesc_task_evaluation.sh   # evaluation
├── src/
│   ├── llm_service.py                   # multi-backend VLM client
│   ├── utils.py
│   ├── phase_diagram_understanding_task/obtain_SemDesc_task_result.py
│   └── task_evaluation/
│       ├── SemDesc_task_evaluation.py
│       ├── evaluation_metrics.py
│       ├── llm_as_judge.py
│       └── vlm_json_parsing.py
└── BERTScore_model/README.md
```

## Environment

Python ≥ 3.10.

```bash
pip install tqdm pyyaml rouge-score bert-score json-repair scikit-learn
```

The BERTScore backbone is not shipped; see [`BERTScore_model/README.md`](BERTScore_model/README.md) to download `xlnet-large-cased`.

## Data

| File | Instances | Images |
|---|---|---|
| `task_dataset/phase_evaluation_data_200.json` | 200 | `phase_0001` – `phase_0200` |
| `task_dataset/phase_evaluation_expanded_data_800.json` | 800 | `phase_0201` – `phase_1000` |

Each instance provides `sample_id`, `img_path` (file name inside `phase_images/`), and `associative_matching_result`, which holds `two_stage_llm_integration` (the ground-truth description) and `dimension_multi_classification`. Both splits share the same schema.

## Run

Both scripts are configured through the variables at the top of each file. **All required values are empty by default** and must be filled in before running. `BACKEND`, `TEST_MODEL` and `LLM_MODEL` are validated at startup; the remaining empty values are passed to the CLI as-is.

### 1. Inference — `scripts/run_SemDesc_task.sh`

| Variable | Meaning |
|---|---|
| `INPUT_LIST` | task JSON files to run |
| `OUTPUT_DIR` | directory for the prediction JSON |
| `BACKEND` | `local` / `dashscope` / `jiekou` / `zhipu` / `tencent` |
| `<BACKEND>_MODEL` | model name, e.g. `JIEKOU_MODEL`, `DASHSCOPE_MODEL` |
| `<BACKEND>_BASE_URL` | endpoint URL, used by `local` / `dashscope` / `tencent` |
| `MAX_WORKERS`, `TEMPERATURE`, `TOP_P` | concurrency and sampling |

API keys come from the environment and are never written into the script: `LOCAL_API_KEY`, `DASHSCOPE_API_KEY`, `JIEKOU_API_KEY`, `ZHIPU_API_KEY`, `TENCENT_API_KEY`.

```bash
export JIEKOU_API_KEY="<your-key>"
# fill INPUT_LIST, OUTPUT_DIR, BACKEND, JIEKOU_MODEL, MAX_WORKERS, TEMPERATURE, TOP_P
bash scripts/run_SemDesc_task.sh
```

`DESCRIPTION_MODE` selects the prompting strategy: `specified_dimension` | `unlimited_description` (default) | `one_shot_description` | `three_shot_description`.

### 2. Evaluation — `scripts/run_SemDesc_task_evaluation.sh`

| Variable | Meaning |
|---|---|
| `TEST_MODEL` | model name used in the prediction file names |
| `PREDICTION_DIR`, `PREDICTION_FILES` | prediction JSON files to score |
| `OUTPUT` | metrics output file |
| `METRIC_MODE` | `traditional` / `sample_details` / `llm_as_judge` (default) / `numerical_prediction_error` / `multi_label_recall_error` |
| `LLM_MODEL` | judge model, required when `METRIC_MODE=llm_as_judge` |
| `LLM_BACKEND` | judge backend, empty falls back to the evaluator default |

Optional environment variables: `CONDA_ENV` (conda env to activate), `CUDA_VISIBLE_DEVICES` (default `0`), `BERTSCORE_DEVICE` (default `cuda:0`), `LLM_API_KEY` (judge key).

```bash
export LLM_API_KEY="<your-key>"
# fill TEST_MODEL, PREDICTION_DIR, PREDICTION_FILES, OUTPUT, LLM_MODEL
bash scripts/run_SemDesc_task_evaluation.sh
```

`traditional` reports ROUGE, BERTScore, numerical prediction error and multi-label recall; `llm_as_judge` scores each description with the rubric in `prompts/semantic_quality_score.yaml` over the five dimensions listed in `DIMENSION_LABELS`. An interrupted judge run can be resumed by setting `LLM_REPROCESS_FAILED=true` and pointing `LLM_REPROCESS_FAILED_FROM` at the previous output.

## BERTScore backbone

`BERTScore_model/xlnet-large-cased` is not included. See [`BERTScore_model/README.md`](BERTScore_model/README.md) for the official download path.
