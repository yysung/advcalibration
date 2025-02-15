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



def merge_categories_all(category):
    category_merges = {
        "Other Arts": "Other Fine Arts",
        "Geo/CE/other": "Geo/CE",
        "Myth": "Myth/Other Academic",
        "World/Other": "World/Other Lit"
    }

    category = category_merges[category] if category in category_merges else category
    if "Arts" in category or "Music" in category or "Painting" in category:
        return "Arts"
    elif "Hist" in category:
        return "History"
    elif "Lit" in category:
        return "Literature"
    elif "Religion" in category or "Social Science" in category or "Myth" in category or "Philosophy" in category:
        return "RMPSS"
    elif "Bio" in category or "Physics" in category or "Chem" in category or "Sci" in category:
        return "Science"
    else:
        return category

df_list = []
category_stats = {}
error_keys = []
question_per_packet = []
human_per_question = {}
packet_team = defaultdict(lambda: {})
for packet in range(1, 13):
    print(f"Processing packet {packet}...")
    packet_file = json.load(open(f'./model_human_output/packet{packet}_model_human_output.json', 'r'))
    packet_file_with_cats = json.load(open(f'./buzzpoints_merged/packet{packet}_buzzpoints_merged.json', 'r'))

    # Find categories for each index in indices, from cat_data
    categories = {}
    for q in packet_file_with_cats:
        categories[q['tossup_index']] = q['category']

    packet_team_list = []
    for data, cat_data in zip(packet_file, packet_file_with_cats):
        print(data['tossup_index'])
        packet_df = calculate_team_buzz_rates(data)
        packet_df['Packet'] = packet
        packet_df.reset_index(inplace=False)
        
        df_list.append(packet_df)
        for item in packet_df.Team.tolist():
            packet_team_list.append(item)

        try:
            category = categories[data['tossup_index']]
            category = merge_categories_all(category)
            # if category in category_merges:
            #     category = category_merges[category]
            if category not in category_stats:
                category_stats[category] = []
            category_stats[category].append(packet_df)
        except KeyError:
            error_keys.append(data['tossup_index'])

        human_per_question = {}
        human_per_question['tossup_index'] = data['tossup_index']
        human_per_question['packet'] = packet
        question_per_packet.append(human_per_question)
    packet_team[packet] = set(packet_team_list)

all_combined_df = pd.concat(df_list)
for cat, data in category_stats.items():
    category_stats[cat] = pd.concat(category_stats[cat])
for row in question_per_packet:
    row['total_human_teams'] = list(packet_team[row['packet']])

# For each team, get their incorrect and correct buzz rates for each category
team_category_stats = {}
for cat in category_stats.keys():
    team_category_stats[cat] = {}
    for team in all_combined_df.Team.unique():

        # Sum the correct buzzes and incorrect buzzes for each category
        if team not in team_category_stats[cat]:
            team_category_stats[cat][team] = {}
        team_category_stats[cat][team]["Correct Buzzes"] = category_stats[cat][category_stats[cat].Team == team].groupby('Team')["Correct Buzzes"].sum()
        team_category_stats[cat][team]["Incorrect Buzzes"] = category_stats[cat][category_stats[cat].Team == team].groupby('Team')["Incorrect Buzzes"].sum()
        team_category_stats[cat][team]["Total Runs"] = category_stats[cat][category_stats[cat].Team == team].groupby('Team')["Total Runs"].sum()
        team_category_stats[cat][team]["Incorrect Buzz Rate"] = team_category_stats[cat][team]["Incorrect Buzzes"] / team_category_stats[cat][team]["Total Runs"]
        team_category_stats[cat][team]["Correct Buzz Rate"] = team_category_stats[cat][team]["Correct Buzzes"] / team_category_stats[cat][team]["Total Runs"]
        team_category_stats[cat][team]["Buzz Margin"] = team_category_stats[cat][team]["Correct Buzzes"] / team_category_stats[cat][team]["Incorrect Buzzes"]

# Print categories ordered by buzz margin
def sort_and_print(cats, name):
    print("\n\n*** " + name + " Buzz Margins by Category ***")
    sorted_cats = {k: v for k, v in sorted(cats.items(), key=lambda item: item[1], reverse=True)}
    for cat, margin in sorted_cats.items():
        print(f"{cat}: {margin}")

# Print in a tabular format
def print_formatted(stats):
    headers = ["Correct Buzzes", "Incorrect Buzzes", "Total Runs", "Incorrect Buzz Rate", "Correct Buzz Rate", "Buzz Margin"]

    all_human_stats = {}
    for category, category_info in stats.items():
        human_sums = [0] * len(headers)
        print("\n\n*** Category: ", category, " ***")
        print("\n", "Team\t\t", "\t".join(headers))
        print("-----------------------------------------------------------------------------------------------\n")
        for team in all_combined_df.Team.unique():
            print(team, end="\t\t")
            for i, header in enumerate(headers):
                try:
                    print(category_info[team][header].values[0].round(4), end="\t\t")
                    if "H" in team:
                        human_sums[i] += category_info[team][header].values[0]
                except IndexError:
                    print("0*", end="\t\t")

        # Also do H1 through H18 combined (so, sum all rows of category_info where "team" starts with H)
        print("Human Sums: ", end="\t")
        for i, header in enumerate(headers):
            if header == "Buzz Margin":
                print((human_sums[0]/human_sums[1]).round(4), end="\t\t")
            else:
                print(human_sums[i].round(4), end="\t\t")
            all_human_stats[category] = (human_sums[0]/human_sums[1]).round(4)

    # Order the categories by buzz margin (descending) for M1
    m1_cats = {}
    m2_cats = {}
    m3_cats = {}
    for category in stats:
        for model, model_vals in {"M1": m1_cats, "M2": m2_cats, "M3": m3_cats}.items():
            try:
                model_vals[category] = stats[category][model]["Buzz Margin"].values[0]
            except IndexError:
                model_vals[category] = 0

    sort_and_print(m1_cats, "M1")
    sort_and_print(m2_cats, "M2")
    sort_and_print(m3_cats, "M3")
    sort_and_print(all_human_stats, "Human")

print_formatted(team_category_stats)

