import pandas as pd
import numpy as np 
import json
import pdb
import pickle
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
scaler = MinMaxScaler()


with open("./../data/questions_packets/human_team_per_question.pkl", "rb") as file:
    team_per_question = pickle.load(file)
tpq_df = pd.DataFrame(team_per_question)

def calculate_SHq(question):
    
    results = {}
    total_humans = 0
    current_h_t = 0 
    
    for model_key in question:
        if model_key.startswith('M'):  
            SHq = 0
            propagated = {}
            for run in question[model_key]:
                if 'position_items' in run:
                    position_items = run.get("position_items",{}) 
                elif 'position' in run:
                    position_items = run.get("position",{}) 
                else: 
                    position_items = {}
                human_items = {
                    key: value for key, value in position_items.items() if key.startswith("H")
                }

                total_humans = len(tpq_df[tpq_df['tossup_index']==question['tossup_index']]['total_human_teams'].values[0]) # total number of humans who saw the question
                propagated = propagated | human_items

                correct_humans = sum(1 for value in propagated.values() if value == "+")

                current_h_t = correct_humans / total_humans if total_humans > 0 else 0
                
                g_t = 1 if run['correctness'] else -1
                c_t = np.exp(run['conf'])  

                SH_run = (1 - current_h_t) * c_t * g_t
                SHq += SH_run
                
            results[model_key] = SHq

    return results


def compute_human_adjusted_score(gt, ct, ht):

    T = len(gt)  # Total timesteps
    bt = np.zeros(T)  # Buzz probabilities
    bege = np.zeros(T)  # bt * gt for each timestep
    SH_q = 0.0

    for t in range(T):
        prod = 1.0
        for i in range(t):
            prod *= (1 - ct[i])
        bt[t] = ct[t] * prod
    
    # Compute bege = bt * gt
    for t in range(T):
        bege[t] = bt[t] * gt[t]
    
    # Compute SH_q for cases where humans did not buzz
    SH_prime_q = sum(bege)
    
    # Probability of system buzzing before humans
    prob_system_buzz_before_humans = sum(ht[t] * sum(bege[:t+1]) for t in range(T))
    
    # Probability of humans not buzzing
    prob_humans_not_buzz = 1 - sum(ht)

    SH_q = prob_system_buzz_before_humans + (prob_humans_not_buzz * SH_prime_q)
    
    return SH_q


def calculate_SHq_comp(question):
    results = {'M1':[],
          'M2':[],
          'M3':[],
          'M4':[],
          'M5':[],
          'M6':[],}
    for model_key in question:
        if model_key.startswith('M'):  
            g_t_q = []
            c_t_q = []
            h_t_q = []

            SHq_comp = 0
            for run in question[model_key]:
                
                propagated = {}
                if 'position_items' in run:
                    position_items = run.get("position_items",{}) 
                elif 'position' in run:
                    position_items = run.get("position",{}) 
                else: 
                    position_items = {}

                human_items = {
                    key: value for key, value in position_items.items() if key.startswith("H")
                }
                total_humans = len(tpq_df[tpq_df['tossup_index']==question['tossup_index']]['total_human_teams'].values[0]) # total number of humans who saw the question

                propagated = propagated | human_items
                correct_humans = sum(1 for value in propagated.values() if value == "+")
                current_h_t = correct_humans / total_humans if total_humans > 0 else 0
                g_t = 1 if run['correctness'] else 0
                c_t = np.exp(run['conf'])  
                g_t_q.append(g_t)
                c_t_q.append(c_t)
                h_t_q.append(current_h_t)


            human_score_per_question = compute_human_adjusted_score(g_t_q, c_t_q, h_t_q)
            results[model_key]=human_score_per_question
    
    return results

def min_max_scale_invert(metric_result): 
    
    df = pd.DataFrame(metric_result)
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    #df_inverted = df_scaled.map(lambda value: 1 / (1 + value))
    df_inverted = df_scaled.map(lambda value: 1-value)
    return df_inverted

def scale_invert(metric_result): 
    
    df = pd.DataFrame(metric_result)
    df_inverted = df.map(lambda value: 1 / (1 + value))
    return df_inverted

def calculate_ECE(confidence_accuracy_pairs, num_bins=10):

    bin_edges = np.linspace(0, 1, num_bins + 1)
    bin_totals = np.zeros(num_bins)
    bin_confidences = np.zeros(num_bins)
    bin_accuracies = np.zeros(num_bins)

    for conf, acc in confidence_accuracy_pairs:
        bin_idx = np.digitize(conf, bin_edges, right=True) - 1
        bin_totals[bin_idx] += 1
        bin_confidences[bin_idx] += conf
        bin_accuracies[bin_idx] += acc

    ece = 0
    for i in range(num_bins):
        if bin_totals[i] > 0:  
            avg_conf = bin_confidences[i] / bin_totals[i]
            avg_acc = bin_accuracies[i] / bin_totals[i]
            ece += (bin_totals[i] / len(confidence_accuracy_pairs)) * abs(avg_conf - avg_acc)
    
    return ece

def calculate_ECE_bulk(packets, elicit, num_bins=10):
    #print('ece')
    ece_results = {
        "M1": [],
        "M2": [],
        "M3": [],
        "M4": [],
        "M5": [],
        "M6": []
    }
    conf_acc_model = {
        "M1": [],
        "M2": [],
        "M3": [],
        "M4": [],
        "M5": [],
        "M6": []
    }
    
    for data in packets:
        for model_key in data.keys():
            if model_key.startswith('M'):  
                last_run = data[model_key][-1]
                if elicit=='logit':
                    conf = np.exp(last_run['conf'])
                else:
                    conf = last_run['conf']
                acc = 1 if last_run['correctness'] else 0
                conf_acc_model[model_key].append((conf,acc))
    
    
    for model in conf_acc_model:
        ece_results[model] = calculate_ECE(conf_acc_model[model], num_bins)
        #print(model, len(conf_acc_model[model]))
    return ece_results

def calculate_ECE_bulk_by_run(packets, elicit, num_bins=10):
    #print('ece')
    ece_results = {
        "M1": [],
        "M2": [],
        "M3": [],
        "M4": [],
        "M5": [],
        "M6": []
    }
    conf_acc_model = {
        "M1": [],
        "M2": [],
        "M3": [],
        "M4": [],
        "M5": [],
        "M6": []
    }
    
    for data in packets:
        for model_key in data.keys():
            if model_key.startswith('M'):
                for run in data[model_key]:
                    if elicit=='logit':
                        conf = np.exp(run['conf'])
                    else:
                        conf = run['conf']
                    acc = 1 if run['correctness'] else 0
                    conf_acc_model[model_key].append((conf,acc))
    #print(conf_acc_model)
    
    for model in conf_acc_model:
        #print(model, len(conf_acc_model[model]))
        ece_results[model] = calculate_ECE(conf_acc_model[model], num_bins)
        
    return ece_results


def calculate_brier_score(confidence_accuracy_pairs):
    
    brier_score_sum = 0
    for conf, acc in confidence_accuracy_pairs:
        brier_score_sum += (conf - acc) ** 2
    return brier_score_sum / len(confidence_accuracy_pairs)

def calculate_brier_score_bulk(packets, elicit):
    #print('brier')
    brier_results = {
        "M1": [],
        "M2": [],
        "M3": [],
        "M4": [],
        "M5": [],
        "M6": []
    }
    conf_acc_model = {
        "M1": [],
        "M2": [],
        "M3": [],
        "M4": [],
        "M5": [],
        "M6": []
    }
    
    for data in packets:
        for model_key in data.keys():
            if model_key.startswith('M'):  
                last_run = data[model_key][-1]
                if elicit=='logit':
                    conf = np.exp(last_run['conf'])
                else:
                    conf = last_run['conf']
                acc = 1 if last_run['correctness'] else 0
                conf_acc_model[model_key].append((conf,acc))
    
    
    for model in conf_acc_model:
        #print(model, len(conf_acc_model[model]))
        brier_results[model] = calculate_brier_score(conf_acc_model[model])
        
    return brier_results

def calculate_brier_score_bulk_by_run(packets, elicit):
    #print('brier')
    brier_results = {
        "M1": [],
        "M2": [],
        "M3": [],
        "M4": [],
        "M5": [],
        "M6": []
    }
    conf_acc_model = {
        "M1": [],
        "M2": [],
        "M3": [],
        "M4": [],
        "M5": [],
        "M6": []
    }
    
    for data in packets:
        for model_key in data.keys():
            if model_key.startswith('M'):
                for run in data[model_key]:
                    if elicit=='logit':
                        conf = np.exp(run['conf'])
                    else:
                        conf = run['conf']
                    acc = 1 if run['correctness'] else 0
                    conf_acc_model[model_key].append((conf,acc))
    
    for model in conf_acc_model:
        #print(model, len(conf_acc_model[model]))
        brier_results[model] = calculate_brier_score(conf_acc_model[model])
        
    return brier_results


