# MatPhaseBench: A Semantics-Guided Benchmark for Materials Phase Diagrams Understanding

## Abstract

Materials phase diagrams are a core knowledge representation in materials science, encoding temperature, composition, phase stability, and phase transformation pathways, with their full understanding requiring thermodynamic mechanism analysis and scientific reasoning. Although VLMs have shown promise in scientific image understanding, their systematic evaluation on such logically complex images demanding deep mechanistic interpretation remains limited, and phase diagrams provide a challenging testbed for this purpose. We introduce MatPhaseBench, a high-quality, high-reliability benchmark for complex scientific image understanding, focused on materials phase diagrams. MatPhaseBench is constructed from 3,681 phase-diagram-related papers in classical materials science journals, from which 200 high-quality diagram-text pairs were selected, covering 189 material systems and 70 elements. The benchmark has three key features: (1) targeting complex scientific image understanding—it moves beyond simple objective tests to open-ended tasks requiring deep comprehension; (2) comprehensive image-text alignment—semantic information directly associated with images is fully preserved during literature mining and matching; (3) high-quality human-supervised text acquisition—all descriptions undergo strict manual validation. Experimental results show that current VLMs remain substantially behind expert-level understanding: they are largely limited to surface visual perception, lack deep reasoning grounded in thermodynamic mechanisms, have limited domain awareness and expert analytical experience, and perform poorly in distinguishing fine-grained differences in composite or multi-diagram settings. Overall, MatPhaseBench constitutes a challenging research-grade benchmark, providing a foundational platform for complex scientific image understanding, phase diagram analysis, and trustworthy multimodal AI in science.

## Environment

Create and activate a Python environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you use GPU-based BERTScore, install a PyTorch build compatible with your CUDA version.


## Dataset

The benchmark data are stored in:

```text
dataset/MatPhaseBench.json
```

Each sample contains fields such as:

- `sample_id`: unique sample identifier.
- `image_path`: relative path to the phase diagram image.
- `img_group`: image grouping identifier.
- `material_info.chemical_information.elements`: chemical elements involved in the sample.
- `material_info.chemical_information.systems`: chemical systems involved in the sample.
- `sample_title`: phase diagram caption text.
- `ground_truth`: The ground truth description of the phase diagram.
- `dimension_multi_classification.labels`: semantic-dimension labels for the sample.

Images are stored under:

```text
dataset/images/
```



## Usage 

### VLM's prediction of phase diagrams

Use:

```bash
bash scripts/run_MatPhaseBench_task.sh
```


### Evaluation of VLM's prediction

Use:

```bash
bash scripts/run_MatPhaseBench_evaluation.sh
```

### BERTScore Model

The evaluation script expects an XLNet model directory and baseline TSV file:

```bash
BERTSCORE_XLNET_MODEL="${PROJECT_ROOT}/BERTScore_model/xlnet-large-cased"
BERTSCORE_XLNET_BASELINE_PATH="${BERTSCORE_XLNET_MODEL}/xlnet-large-cased.tsv"
```

Replace these paths if your BERTScore model is stored elsewhere.

The option:

```bash
--bertscore-rescale-with-baseline
```

enables BERTScore baseline rescaling, which maps raw BERTScore values to a more interpretable scale using the provided baseline file.

