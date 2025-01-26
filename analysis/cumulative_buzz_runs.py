
import pickle
import re
import pandas as pd
import numpy as np
import json 
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import pdb
import seaborn as sns


import pickle
import re
with open("./../data/questions_packets/human_model_team_per_question.pkl", "rb") as file:
    team_per_question = pickle.load(file)
with open("./../data/questions_packets/all_teams.pkl", "rb") as file:
    all_teams = pickle.load(file)

run_stats={}
for team in all_teams:
    run_stats[team] =  defaultdict(lambda: {"run": run_index, "seen": 0, "buzz": 0, "buzz_correctness": 0, "buzz_incorrectness": 0})

for packet in range(1, 13):
    #print('packet', packet)
    data = json.load(open(f'./model_human_output/packet{packet}_model_human_output.json', 'r'))
    for q_idx, question in enumerate(data):
        
        # team_tmp = [item[1][0] for item in question['position'].items()]
        # teams_participated = [re.search(r'\[(\w+\d+),', item).group(1) for item in team_tmp]
        human_teams_participated = [team for team in team_per_question[question['tossup_index']]['seen'] if team.startswith('H')]
        model_teams_participated = [team for team in team_per_question[question['tossup_index']]['seen'] if team.startswith('M')]
        
        first_model = question['M1']
        for run_index, run in enumerate(first_model):
            buzz_opp = len(first_model)
            run_text = f"Run {run_index}"
            for team in human_teams_participated:
                run_stats[team][run_text]["seen"]+= 1
            for team in model_teams_participated:
                run_stats[team][run_text]["seen"]+= 1
            
            if 'position_items' in run:
                buzz_data = run['position_items']
                teams_buzzed = list(run['position_items'].keys())
                
                human_teams_buzzed = [team for team in teams_buzzed if team.startswith('H')]
                for human_key in human_teams_buzzed:
                    #print(human_key)
                    run_stats[human_key][run_text]["buzz"] +=1
                    # number of + buzz in each run
                    human_with_correctness = [key for key, value in run['position_items'].items() if value == '+' and key in human_teams_buzzed]
                    human_with_incorrectness = [key for key, value in run['position_items'].items() if value == '-' and key in human_teams_buzzed]
                    for human_key in human_with_correctness:
                        run_stats[human_key][run_text]["buzz_correctness"]+= 1
                    for human_key in human_with_incorrectness:
                        run_stats[human_key][run_text]["buzz_incorrectness"]+= 1

                model_teams_buzzed = [team for team in teams_buzzed if team.startswith('M')]
                for model_key in model_teams_buzzed:
                    # print(model_key)
                    run_stats[model_key][run_text]["buzz"] += 1
                    # number of + buzz in each run
                    model_with_correctness = [key for key, value in run['position_items'].items() if value == '+' and key in model_teams_buzzed]
                    model_with_incorrectness = [key for key, value in run['position_items'].items() if value == '-' and key in model_teams_buzzed]
                    for model_key in model_with_correctness:
                        run_stats[model_key][run_text]["buzz_correctness"] += 1
                    for model_key in model_with_incorrectness:
                        run_stats[model_key][run_text]["buzz_incorrectness"] += 1
                
            else:
                pass
        


runs = list(range(0, 13))

team_correct_counts = {team: [] for team in run_stats.keys()}
team_incorrect_counts = {team: [] for team in run_stats.keys()}
team_run_counts = {team: [] for team in run_stats.keys()}

for team, team_data in run_stats.items():
    for run in runs:
        run_text = f"Run {run}"
        team_correct_counts[team].append(team_data[run_text]["buzz_correctness"])
        team_incorrect_counts[team].append(team_data[run_text]["buzz_incorrectness"])
        team_run_counts[team].append(team_data[run_text]["seen"])

team_cumulative_correct_counts = {team: np.cumsum(counts) for team, counts in team_correct_counts.items()}
team_cumulative_incorrect_counts = {team: np.cumsum(counts) for team, counts in team_incorrect_counts.items()}
team_cumulative_run_counts = {team: np.cumsum(counts) for team, counts in team_run_counts.items()}

human_teams = [team for team in team_correct_counts.keys() if not team.startswith("M")]
model_teams = [team for team in team_correct_counts.keys() if team.startswith("M")]

human_total_correct = {team: sum(counts) for team, counts in team_cumulative_correct_counts.items() if team in human_teams}
human_quartiles = np.percentile(list(human_total_correct.values()), [25, 50, 75])

def assign_quartile(total_correct):
    if total_correct <= human_quartiles[0]:
        return 0
    elif total_correct <= human_quartiles[1]:
        return 1
    elif total_correct <= human_quartiles[2]:
        return 2
    else:
        return 3

quartile_groups = {i: [] for i in range(4)}
for team, total_correct in human_total_correct.items():
    quartile = assign_quartile(total_correct)
    quartile_groups[quartile].append(team)

print('quartile_groups', quartile_groups)

quartile_cumulative_correct_counts = {
    i: np.array([
        sum(team_cumulative_correct_counts[team][run_idx] for team in teams) / 
        max(sum(team_cumulative_run_counts[team][run_idx] for team in teams), 1)
        for run_idx in range(len(runs))
    ])
    for i, teams in quartile_groups.items()
}

quartile_cumulative_incorrect_counts = {
    i: np.array([
        sum(team_cumulative_incorrect_counts[team][run_idx] for team in teams) / 
        max(sum(team_cumulative_run_counts[team][run_idx] for team in teams), 1)
        for run_idx in range(len(runs))
    ])
    for i, teams in quartile_groups.items()
}

model_cumulative_correct_counts = {
    team: team_cumulative_correct_counts[team] / np.maximum(team_cumulative_run_counts[team], 1)
    for team in model_teams
}
model_cumulative_incorrect_counts = {
    team: team_cumulative_incorrect_counts[team] / np.maximum(team_cumulative_run_counts[team], 1)
    for team in model_teams
}



# Plot Noramlized Cumulative Buzz Rates
adjusted_runs = [run + 1 for run in runs]

quartile_colors = list(mcolors.LinearSegmentedColormap.from_list("blue_gradient", ["lightblue", "navy"])(np.linspace(0, 1, 4)))
model_colors = ["green", "orange", "purple"]
model_team_color_map = {team: model_colors[i] for i, team in enumerate(model_teams)}

legend_name_mapping = {
    "M1": "GPT-4",
    "M2": "GPT-4o",
    "M3": "Mistral-7b-Instruct"
}

line_thickness = 3
axis_label_fontsize = 30
tick_label_fontsize = 30
legend_fontsize = 20
fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

# Human Quartiles
for quartile, correct_counts in quartile_cumulative_correct_counts.items():
    label = f"Human Quartile {quartile + 1}"
    axes[0].plot(adjusted_runs, correct_counts, marker='o', label=label, color=quartile_colors[quartile], linewidth=line_thickness)

# Model Teams
for team, correct_counts in model_cumulative_correct_counts.items():
    axes[0].plot(adjusted_runs, correct_counts, marker='x', label=legend_name_mapping.get(team, team), color=model_team_color_map[team], linewidth=line_thickness)

axes[0].set_title('Correct Buzzes', fontsize=35)
axes[0].set_xlabel('Seen Clues', fontsize=axis_label_fontsize)
axes[0].set_ylabel('Cumulative Buzzes (Normalized)', fontsize=axis_label_fontsize)
axes[0].grid(visible=True, linestyle='--', linewidth=0.5)
axes[0].tick_params(axis='x', labelsize=tick_label_fontsize)
axes[0].tick_params(axis='y', labelsize=tick_label_fontsize)

axes[0].legend().set_visible(False)

# Incorrect Buzzes for Human Quartiles
for quartile, incorrect_counts in quartile_cumulative_incorrect_counts.items():
    label = f"Human Quartile {quartile + 1}"
    axes[1].plot(adjusted_runs, incorrect_counts, marker='o', label=label, color=quartile_colors[quartile], linewidth=line_thickness)

# Incorrect Buzzes for Model Teams
for team, incorrect_counts in model_cumulative_incorrect_counts.items():
    axes[1].plot(adjusted_runs, incorrect_counts, marker='x', label=legend_name_mapping.get(team, team), color=model_team_color_map[team], linewidth=line_thickness)

axes[1].set_title('Incorrect Buzzes', fontsize=35)
axes[1].set_xlabel('Seen Clues', fontsize=axis_label_fontsize)
axes[1].grid(visible=True, linestyle='--', linewidth=0.5)
axes[1].tick_params(axis='x', labelsize=tick_label_fontsize)
axes[1].tick_params(axis='y', labelsize=tick_label_fontsize)

axes[1].legend(fontsize=legend_fontsize, loc='upper right')

plt.tight_layout()
plt.show()
