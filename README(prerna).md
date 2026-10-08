# UR2PhD Starter Assignment: SemEval-2027 Task 2 (DiCo-NLI), Track 1
## Results

| System | Weighted F1 | SoftCons | HardCons |
| --- | --- | --- | --- |
| Majority baseline (all FORWARD_ENTAILMENT) | 0.129 | 0.000 | 0.000 |
| A: DeBERTa-v3-base, mean ± std over 3 seeds | 0.826 ± 0.014 | 0.868 ± 0.002 | 0.821 ± 0.009 |
| B: Qwen2.5-1.5B-Instruct, zero-shot | 0.287 | 0.126 | 0.090 |
| B: Qwen2.5-1.5B-Instruct, few-shot (8 examples) | 0.333 | 0.271 | 0.181 |

System A per seed:

| Seed | Weighted F1 | SoftCons | HardCons |
| --- | --- | --- | --- |
| 13 | 0.836 | 0.866 | 0.830 |
| 42 | 0.810 | 0.870 | 0.812 |
| 123 | 0.830 | 0.866 | 0.819 |

The standard deviation is the sample standard deviation over the three seeds.

## Files I added

| File | What it does |
| --- | --- |
| `baseline.py` | Part 0. Writes the majority-baseline predictions. |
| `part1.ipynb` | Part 1. Label distribution, three examples per label, how reversed pairs are linked. |
| `system_a_deberta.ipynb` | Part 2. Fine-tunes DeBERTa-v3-base with three seeds, predicts on dev, scores. |
| `system_b_llm.ipynb` | Part 3. Prompts Qwen2.5-1.5B-Instruct zero-shot and few-shot, parses, scores. |
| `part4_analysis.ipynb` | Part 4. Confusion matrices, error lists, pairs that fail SoftCons. |
| `report.pdf` | The report. |

Prediction files (format `instance_id,label`) in `results/`:

| System | Predictions |
| --- | --- |
| Majority baseline | `results/majority_dev.csv` |
| DeBERTa seed 13 / 42 / 123 | `results/deberta_seed13_dev.csv`, `results/deberta_seed42_dev.csv`, `results/deberta_seed123_dev.csv` |
| Qwen zero-shot | `results/qwen_zeroshot_dev.csv` |
| Qwen few-shot | `results/qwen_fewshot_dev.csv` |

Also in `results/`: a `*_scores` folder per system with the official scorer output, `deberta_per_seed.csv`, `deberta_summary.csv`, `qwen_summary.csv`, and `qwen_zeroshot_raw.csv` / `qwen_fewshot_raw.csv` with the model's raw text output next to each pair.
## How to reproduce : The scorer needs Python 3.10 or newer. Run everything from the repository root.
### Part 0: majority baseline (any machine, no GPU)
```bash
python3 baseline.py
python3 -m evaluation_functions \
  --gold final_data/dev/dico_nli_dev_track1_reference.csv \
  --predictions results/majority_dev.csv \
  --output-dir results/majority_scores
```
FORWARD_ENTAILMENT and BACKWARD_ENTAILMENT tie as the most frequent train label (881 each). I used FORWARD_ENTAILMENT.
### Part 1: data exploration (no GPU) - Open `part1.ipynb` and run all cells. Needs `pandas`.
### Part 2: System A (Google Colab, T4 GPU)
Open `system_a_deberta.ipynb` in Colab, set the runtime to T4 GPU, and run all cells. The first cell clones this repository. Training takes about 2 minutes per seed.

| Setting | Value |
| --- | --- |
| Model | `microsoft/deberta-v3-base`, `AutoModelForSequenceClassification`, 4 labels |
| Input | `tokenizer(text1, text2)`, truncation, max length 64 |
| Seeds | 13, 42, 123 |
| Learning rate | 2e-5 |
| Train batch size | 16 |
| Epochs | 5 |
| Weight decay | 0.01 |
| Warm-up steps | 96 |
| Precision | float32 (`model.float()` after loading; see below) |
| Model selection | None. Fixed epoch count; dev is used only for the final scoring. |

### Part 3: System B (Google Colab, T4 GPU)
Open `system_b_llm.ipynb` in Colab, set the runtime to T4 GPU, and run all cells. Each variant takes about 3 minutes.
| Setting | Value |
| --- | --- |
| Model | `Qwen/Qwen2.5-1.5B-Instruct`, loaded in half precision for inference |
| Decoding | Greedy (`do_sample=False`), `max_new_tokens=16` |
| Seeds | Not applicable: greedy decoding is deterministic |
| Few-shot examples | 8 from the train split, 2 per label, `train.groupby("label").sample(2, random_state=0)`, then shuffled with `random_state=0`; given as user/assistant chat turns |
| Parser | Upper-case the output and look for one keyword per label (`EQUIVALEN`, `FORWARD`, `BACKWARD`, `NEGATIVE`). Exactly one keyword found: that label. Otherwise unparsed. |
| Unparsed outputs | Would be assigned NEGATIVE_OTHER. There were 0 of 660 in both variants. |

System prompt (used for both variants):

```text
You are given two short phrases, text1 and text2. Decide how their meanings are related and answer with exactly one of these four labels:

EQUIVALENCE: text1 and text2 mean the same thing.
FORWARD_ENTAILMENT: text1 entails text2. text1 is more specific, so if text1 is true then text2 must be true.
BACKWARD_ENTAILMENT: text2 entails text1. text2 is more specific, so if text2 is true then text1 must be true.
NEGATIVE_OTHER: any other relation. The phrases may be similar, related, opposite, or unrelated, but neither entails the other.

Answer with the label only and nothing else.
```

Each item is sent as a user message of the form `text1: ...` / `text2: ...` on two lines.
The 8 few-shot examples selected by the fixed seed:
| Label | text1 | text2 |
| --- | --- | --- |
| NEGATIVE_OTHER | is jumping | riding |
| EQUIVALENCE | three goals scored by Ronaldo | Ronaldo 's hat trick |
| BACKWARD_ENTAILMENT | with a bracelet a bikini top and jeans | jeans , a dark bikini top , and a bracelet |
| NEGATIVE_OTHER | underwater water | in very clear blue water |
| EQUIVALENCE | between Raffles Place and Marina Bay | linking the Raffles Place district to the Marina Bay area |
| BACKWARD_ENTAILMENT | A boy | A young boy |
| FORWARD_ENTAILMENT | Two grey dogs | Two dogs |
| FORWARD_ENTAILMENT | ` Blade Runner ' Pistorius | Pistorius |

### Part 4: analysis (no GPU)
Open `part4_analysis.ipynb` and run all cells. It reads the prediction files in `results/` and the dev reference file. The error analysis uses DeBERTa seed 13, the best seed by Weighted F1 and HardCons.
## Runs that did not work
**DeBERTa, first attempt.** My first three runs (seeds 13, 42, 123, same hyperparameters) all scored exactly the majority baseline: Weighted F1 0.129, SoftCons 0, HardCons 0. The training loss became NaN after about step 100 for seeds 13 and 123, and stayed near 1.37 (chance level for four labels) for seed 42. I printed the model's dtype and found that `microsoft/deberta-v3-base` was loading in float16, which is unstable for training. Adding `model = model.float()` after loading fixed it. The reported results are from the fixed runs. I did not keep the failed prediction files; removing the `.float()` line reproduces the failure.
**Qwen, first scoring.** My first scoring of System B gave Weighted F1 0.163 (zero-shot) and 0.176 (few-shot), with 122 and 242 outputs counted as unparsed and no EQUIVALENCE predictions. The parser test for an EQUIVALENCE output was returning `None`, so the parser cell that ran had a bug. I reran both variants in a fresh notebook where the parser is checked by assertions before the run. The reported results are from that rerun (0 unparsed). The first run's files were not kept.

## Notes on the consistency metrics
- SoftCons and HardCons are computed over the 277 reversible dev pairs. Gold NEGATIVE_OTHER items count toward Weighted F1 only.
- In `analysis.ipynb` my own recomputation of SoftCons matches the official scorer (0.8664 for seed 13) only when a pair predicted NEGATIVE_OTHER in both directions is counted as not consistent. For DeBERTa seed 13, 37 pairs fail SoftCons: 20 are predicted NEGATIVE_OTHER both ways and 17 are contradictions between the two directions.

## Rules followed
- Only the official train and dev data were used. Dev was not used for training, model selection, or choosing few-shot examples.

## Use of AI assistants
I used Claude (Anthropic) as an AI assistant throughout this assignment. 
It provided most of the code in the notebooks and explained each step, helped diagnose the two failed runs described above, suggested interpretations of the error analysis, and drafted this README, which I then edited. I ran all experiments myself, checked the outputs against the official scorer, and chose the examples for Part 1.
