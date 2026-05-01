
# Key Findings (auto-generated)

## Headline result
- **Best system:** *Mistral Devstral 2 123B* with **P5 RAG dense (k=3)** → micro F1 = **0.964**
  (pooled across Java, Python, JavaScript, C++).
- Per-model best prompts: **DeepSeek v3.2** → P2 few-shot; **Gemma 3 27B-IT** → P2 few-shot; **Mistral Devstral 2 123B** → P5 RAG dense (k=3).

## RQ1 — Does few-shot beat zero-shot?
- Mean lift of P2 few-shot over P1 across 3 models: **+0.161** F1.
- Few-shot improves all models: yes.

## RQ2 — Does structured / self-correction reasoning help?
- P3 taxonomy-tree mean lift: **-0.051** F1.
- P4 self-verify mean lift:    **-0.271** F1.
- Take-away: structured reasoning tends to hurt;
  self-verify is inconsistent across models.

## RQ3 — Does RAG content matter?
- Dense (k=3) − random (k=2) per model:
  - DeepSeek v3.2: -0.017
  - Gemma 3 27B-IT: +0.016
  - Mistral Devstral 2 123B: +0.031
- Models that benefit from real retrieval: **Gemma 3 27B-IT, Mistral Devstral 2 123B**.
- Models where dense ≈ random: **none** → suggests the smell taxonomy
  in the system prompt already supplies most of the signal these models can exploit; retrieval mainly
  acts as a real evidence channel.

## Per-smell — easiest vs hardest
- **Easiest (highest mean F1, support ≥10):** Temporary Field (0.98), Data Clumps (0.98), Inappropriate Intimacy (0.96), Long Method (0.95), Primitive Obsession (0.94)
- **Hardest (lowest mean F1, support ≥10):**  Parallel Inheritance Hierarchies (0.78), Divergent Change (0.80), Switch Statements (0.80), Long Parameter List (0.81), Refused Bequest (0.81)

## Language difficulty
- python: mean F1 = 0.810
- javascript: mean F1 = 0.764
- java: mean F1 = 0.713
- cpp: mean F1 = 0.651

## Cost / latency
- Best F1-per-second is achieved at *('Mistral Devstral 2 123B', 'p5_rag_dense')* (F1=0.964,
  total elapsed 584 s across 4 langs).

## Recommended setup for production
- **Quality-first:** *Mistral Devstral 2 123B + P5 RAG dense (k=3)*.
- **Latency-first:** the model with highest F1 in P1 zero-shot — `DeepSeek v3.2`
  (F1 = 0.772); skips few-shot/RAG context overhead.
- **Open question for future work:** the long tail of low-support smells (Lazy Class, Comments,
  Data Class, Feature Envy) is unstable across runs — collect more annotations before drawing
  conclusions on those classes.
