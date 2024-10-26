import pandas as pd
import numpy as np
import json 
from collections import defaultdict
import pickle

import matplotlib.pyplot as plt
import seaborn as sns

with open("./../data/questions_packets/human_team_per_question.pkl", "rb") as file:
    team_per_question = pickle.load(file)

Team_per_question = [item['total_human_teams'] for item in team_per_question]

team_set = set()
for teams in Team_per_question:
    for team in teams:
        team_set.add(team) 
all_teams = list(team_set)

models = ['M1','M2', 'M3', 'M4', 'M5','M6']
all_teams = set(all_teams+models)
all_teams.remove('A')
all_teams = list(all_teams)

run_stats={}
for team in all_teams:
    run_stats[team] =  defaultdict(lambda: {"run": run_index, "buzz": 0, "buzz_correctness": 0, "buzz_incorrectness": 0})


for packet in range(1, 13):
    data = json.load(open(f'./model_human_output/packet{packet}_model_human_output.json', 'r'))
    for q_idx, question in enumerate(data):
        first_model = question['M1']

        
        for run_index, run in enumerate(first_model):
            buzz_opp = len(first_model)
            run_text = f"Run {run_index}"
            
            if 'position_items' in run:
                buzz_data = run['position_items']
                for human_key in list(run['position_items'].keys()):
                    if human_key.startswith('H'):
                        run_stats[human_key][run_text]["buzz"] +=1
                human_with_correctness = [key for key, value in run['position_items'].items() if value == '+']
                human_with_incorrectness = [key for key, value in run['position_items'].items() if value == '-']
                for human_key in human_with_correctness:
                    #if human_key.startswith('H'):
                    run_stats[human_key][run_text]["buzz_correctness"]+= 1
                for human_key in human_with_incorrectness:
                    #if human_key.startswith('H'):
                    run_stats[human_key][run_text]["buzz_incorrectness"]+= 1
            

for model_key in ['M4', 'M5', 'M6']:
    for packet in range(1, 13):
        data = json.load(open(f'./model_human_output/packet{packet}_model_human_output.json', 'r'))

        for q_idx, question in enumerate(data):
            model = question[model_key]
            
            for run_index, run in enumerate(model):
                run_text = f"Run {run_index}"

                if run_index > 1:
                    if run['buzzed']:

                        if run['correctness']==True:
                            run_stats[model_key][run_text]["buzz_correctness"] += 1
                        elif run['correctness']==False:
                            run_stats[model_key][run_text]["buzz_incorrectness"] += 1
                        break
print(run_stats)