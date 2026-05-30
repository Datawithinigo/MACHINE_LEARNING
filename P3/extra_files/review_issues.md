# Review Issues (Non-LSTM)

These are issues identified during verification against `comment_laia.md` and the assignment PDF that are **not** related to the LSTM section and therefore were not modified.

## 1. Train/Test Split Deviation from PDF

The PDF asks for a **stratified 80/20 random split** (`random_state=42`), but the notebook uses a **temporal intra-patient split** (first 80% of beats per record → train, last 20% → test). This was a deliberate change following Laia's feedback to eliminate window overlap leakage. The temporal approach is better, but it deviates from the assignment specification.

## 2. Inter-Patient Split as Limitation

Laia suggested mentioning the **inter-patient split** (11 records train, 3 records test) as a limitation / future work. She noted she tried it and "it turns out terrible" but it would be the most clinically realistic approach. This should appear in the report or in a markdown cell as a limitation, as Laia requested.

## 3. Task 4 — Deployment Discussion Missing

The PDF (Task 4) requires: *"Discuss which model you would deploy in a real-time cardiac monitor and why, considering classification performance, latency, interpretability, and the cost of false negatives on ventricular beats."* This discussion is not present in the notebook. A markdown cell addressing these points should be added after the comparison table.
