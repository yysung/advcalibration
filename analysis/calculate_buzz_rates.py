import pandas as pd
import numpy as np
import json 
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

### How often is the model confidently incorrect/correct vs compare with each team

def calculate_buzz_rates(data):

    results = []

    for model_key in data:
        if model_key.startswith('M'):  
            model_runs = data[model_key]
            total_runs = len(model_runs)
            incorrect_buzzes = sum(not run['correctness'] for run in model_runs)
            correct_buzzes = sum(run['correctness'] for run in model_runs)
            incorrect_buzz_rate = incorrect_buzzes / total_runs
            correct_buzz_rate = correct_buzzes / total_runs

            results.append({
                'Model': model_key,
                'Incorrect Buzz Rate': incorrect_buzz_rate,
                'Correct Buzz Rate': correct_buzz_rate,
                'Total Runs': total_runs,
                'Incorrect Buzzes': incorrect_buzzes,
                'Correct Buzzes': correct_buzzes
            })

    return pd.DataFrame(results)

def calculate_team_buzz_rates(data):
    """
    Calculate correct and incorrect buzz rates for human teams using the 'position' field.

    Parameters:
        data (dict): Input data containing question details and model performance.

    Returns:
        pd.DataFrame: A DataFrame with buzz rates for each team or an empty DataFrame if no data exists.
    """
    team_stats = {}  

    #print('tossup_index', data['tossup_index'])
    positions = data.get('position', {})
    model_runs = data['M1']
    if len(positions)!=0:
        for word_index, buzzes in positions.items():
            for buzz in buzzes:
                buzz_cleaned = buzz.strip('[]')
                team, value = buzz_cleaned.split(', ')

                if team not in team_stats:
                    team_stats[team] = {
                        "Team": team,
                        "Correct Buzzes": 0,
                        "Incorrect Buzzes": 0,
                        "Total Runs": len(model_runs),
                        'Incorrect Buzz Rate': 0,
                        'Correct Buzz Rate': 0,
                    }
                
                if value == "+":
                    team_stats[team]["Correct Buzzes"] += 1
                elif value == "-":
                    team_stats[team]["Incorrect Buzzes"] += 1

    else:
        team_stats['A'] = {
            "Team": 'A',
            "Correct Buzzes": 0,
            "Incorrect Buzzes": 0,
            "Total Runs": 0,
            'Incorrect Buzz Rate': 0,
            'Correct Buzz Rate': 0,
        }


    return pd.DataFrame(team_stats).T

df_list = []
question_per_packet = []
human_per_question = {}
packet_team = defaultdict(lambda: {})
for packet in range(1, 13):
    print(f"Processing packet {packet}...")
    packet_file = json.load(open(f'./model_human_output/packet{packet}_model_human_output.json', 'r'))

    packet_team_list = []
    for data in packet_file:
        packet_df = calculate_team_buzz_rates(data)

        packet_df['Packet'] = packet  
        
        packet_df.reset_index(inplace=False)
        
        df_list.append(packet_df)
        for item in packet_df.Team.tolist():
            packet_team_list.append(item)
        
        human_per_question = {}
        human_per_question['tossup_index'] = data['tossup_index']
        human_per_question['packet'] = packet
        question_per_packet.append(human_per_question)
    packet_team[packet] = set(packet_team_list)

all_combined_df = pd.concat(df_list)

for row in question_per_packet:
    row['total_human_teams'] = list(packet_team[row['packet']])
    
print(question_per_packet)
