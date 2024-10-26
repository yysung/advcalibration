import json
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

model_run_stats = {
    "M1": defaultdict(lambda: {"correct": 0, "total": 0, "gc": 0, "num_run": 0}),
    "M2": defaultdict(lambda: {"correct": 0, "total": 0, "gc": 0, "num_run": 0}),
    "M3": defaultdict(lambda: {"correct": 0, "total": 0, "gc": 0, "num_run": 0}),
    "M4": defaultdict(lambda: {"correct": 0, "total": 0, "gc": 0, "num_run": 0}),
    "M5": defaultdict(lambda: {"correct": 0, "total": 0, "gc": 0, "num_run": 0}),
    "M6": defaultdict(lambda: {"correct": 0, "total": 0, "gc": 0, "num_run": 0}),
}

run_len = []
for packet in range(1, 13):
    data = json.load(open(f'./model_human_output/packet{packet}_model_human_output.json', 'r'))
    
    for question in data:
        past_run = 0
        for model_key in ["M1", "M2", "M3", "M4", "M5", "M6"]:
            if model_key in question:
                run_len = len(question[model_key])
                for run_index, run in enumerate(question[model_key]):
                
                    run_percentage = ((run_index + 1) / run_len) * 100
                    run_percentage_interval = int(run_percentage // 10) * 10 
                    percentage_key = f"{run_percentage_interval}"

                    past_run = len(run['run'])
                    correctness = run["correctness"]
                    model_run_stats[model_key][percentage_key]["total"] += 1
                    if correctness:
                        model_run_stats[model_key][percentage_key]["correct"] += 1

                    confidence = np.exp(run['conf'])
                    correctness = 1 if run['correctness'] == True else 0
                    model_run_stats[model_key][percentage_key]["num_run"] += 1
                    model_run_stats[model_key][percentage_key]["gc"] += confidence * correctness



for model_key in ["M1", "M2", "M3", "M4", "M5", "M6"]:
    for run_text in list(model_run_stats['M1'].keys()):
        
        model_run_stats[model_key][run_text]['acc_rate'] = (
            np.round(
                model_run_stats[model_key][run_text]["correct"]
                / model_run_stats[model_key][run_text]["total"],
                2,
            )
            if model_run_stats[model_key][run_text]["total"] > 0
            else 0
        )
    
        model_run_stats[model_key][run_text]['avg_weighted_sum'] = (
            np.round(
                model_run_stats[model_key][run_text]["gc"]
                / model_run_stats[model_key][run_text]["num_run"],
                2,
            )
            if model_run_stats[model_key][run_text]["num_run"] > 0
            else 0
        )
    


models =  ["M1", "M2", "M3", "M4", "M5", "M6"]
x_axis_runs =[f"{i}" for i in range(10, 110, 10)] 
data = []

for model in models:
    for i in x_axis_runs:
        data.append({
            "Model": model,
            "Run": f"{i}",
            "Accuracy Rate": model_run_stats[model][f"{i}"]["acc_rate"],
            "Weighted Sum": model_run_stats[model][f"{i}"]["avg_weighted_sum"]
        })

df = pd.DataFrame(data)

df_melted = pd.melt(
    df,
    id_vars=["Model", "Run"],
    value_vars=["Accuracy Rate", "Weighted Sum"],
    var_name="Metric",
    value_name="Value"
)

accuracy_rate_df = df_melted[df_melted["Metric"] == "Accuracy Rate"]
accuracy_rate_df["Model"] = accuracy_rate_df["Model"].map({
    "M1": "GPT-4",
    "M2": "GPT-4o",
    "M3": "Mistral-7b-Instruct",
    "M4": "LLama-2-70b-Chat",
    "M5": "Llama-3.1-8B-Instruct",
    "M6": "Llama-3.1-70B-Instruct"
})

sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=accuracy_rate_df,
    x="Run",
    y="Value",
    hue="Model",
    marker="o"
)

plt.xlabel("Percentage of Clues Revealed (%)", fontsize=30)  
plt.ylabel("Accuracy Rate", fontsize=30)     
plt.xticks(fontsize=25)  
plt.yticks(fontsize=25) 
plt.ylim(0, 1)
plt.legend(title="Model", fontsize=15)
plt.tight_layout()
plt.show()
