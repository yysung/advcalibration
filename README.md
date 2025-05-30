# LLM Calibration Evaluation 

This repository contains the data and code used in our **LLM Calibration Evaluation** project.

## 📂 Accessing the Data

The folder **`[public]ALL_final_buzzpoints_enc.zip`** contains all human and model buzzpoints, along with our new QA dataset .

- **🔑 Password to Access the Folder:** `buzzbuzz`


This repository provides two scoring functions—**`MCE (Grace_q_non_adjusted)`** and **`CalScore (Grace_q_adjusted)`**—used to evaluate model calibration errors in comparison to humans. These metrics are designed to capture both the model's raw performance and its improvement over human baselines.


## MCE Metric implementation
**Definition**:  
Measures how likely the model is **not** to buzz correctly.

**Formula**:  
`Grace_q_non_adjusted = 1 - E[g * c]`

Where:
- `g` is an indicator (1 if the model answered correctly, 0 otherwise)
- `c` is the model's confidence
- `E[g * c]` is the expected confidence-weighted correctness

This is a baseline measure for model performance **without accounting for human responses**.

## CalScore Metric implementation

**Definition**:  
Measures how much the model improves **over human performance**.

**Formula**:  
`Grace_q_adjusted = 1 - E[(1 - h) * g * c]`

Where:
- `h` is the human correctness probability (based on propagated buzzes)
- `g` is the model correctness
- `c` is the model confidence

By discounting runs where humans already buzzed correctly, this metric rewards models that perform **better than humans**.

- `elicit`: Either `'logit'` (if confidence is in logit space) or `'verb'` (if confidence is verbalized confidence promoted to LLM).

## 💡 Usage Example

```python
import numpy as np
from grace_q import calculate_Grace_q_non_adjusted, calculate_Grace_q_adjusted

# Example data format:
# question = {
#     "M1": [{"correctness": True, "conf": 0.8, "position": {...}}, ...],
#     "position": [{"H1": "[H1] +++"}, ...]
# }

score_non_adjusted = calculate_Grace_q_non_adjusted(question, elicit="raw")
score_adjusted = calculate_Grace_q_adjusted(question, elicit="raw")
