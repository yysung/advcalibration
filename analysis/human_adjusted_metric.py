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

def calculate_Grace_q_non_adjusted(question, elicit):
    """
    Computes the non-adjusted Grace_q score for a given question.
    
    Non-adjusted score: 1 - E[g*c]
    - Represents the expected probability that the model does NOT buzz correctly.
    
    Parameters:
    - question (dict): A dictionary containing model runs.
    
    Returns:
    - dict: Mapping each model to its computed Grace_q_non_adjusted value.
    """
    results = {}

    for model_key in question:
        if model_key.startswith('M'):  # Process model responses only
            gc_sum = 0
            count = 0  # Number of runs

            for run in question[model_key]:
                g_t = 1 if run['correctness'] else 0  # Model correctness indicator (1 if correct, 0 if wrong)
                if elicit=='logit':
                    c_t = np.exp(run['conf'])  # Model confidence (scaled)
                else:
                    c_t = run['conf']

                gc_sum += g_t * c_t  # Expected probability of model being correct and confident
                count += 1

            # Compute expectation
            E_g_c = gc_sum / count if count > 0 else 0
            Grace_q_non_adjusted = 1 - E_g_c  # Final computation

            results[model_key] = Grace_q_non_adjusted

    return results


def calculate_Grace_q_adjusted(question, elicit):
    """
    Computes the adjusted Grace_q score for a given question while propagating correct human responses over runs.
    
    Adjusted score: 1 - E[(1-h) g*c]
    - Represents how much the model does NOT improve over humans.
    
    Parameters:
    - question (dict): A dictionary containing model runs and human responses.
    
    Returns:
    - dict: Mapping each model to its computed Grace_q_adjusted value.
    """
    results = {}
    buzzed_teams = [list(buzzed_team.values())[0] for buzzed_team in question['position']]
    human_buzzed_teams = [team.strip('[]')[:-3] for team in buzzed_teams if team.strip('[]').startswith('H')]
    #print(f"tossup {question['tossup_index']}")
    for model_key in question:
        if model_key.startswith('M'):  # Process model responses only
            hgc_sum = 0
            count = 0  # Number of runs

            # Initialize `propagated` to store cumulative human buzz correctness
            
            propagated = {}

            for i, run in enumerate(question[model_key]):
                # Extract human buzz correctness for this run
                position_items = run.get("position_items", run.get("position", {})) 
                human_items = {key: value for key, value in position_items.items() if key.startswith("H")}

                # Update `propagated` with new human buzzes
                propagated.update(human_items)

                # Debugging output
                #print("propagated:", propagated)

                # # Get total number of humans who saw the question
                # total_humans = len([
                #     team for team in team_per_question.get(question.get('tossup_index'), {}).get('seen', []) 
                #     if team.startswith('H')
                # ])
                # Get total number of humans who buzzed at this question
                total_humans = len(human_buzzed_teams)
                
                # Compute probability of human buzzing correctly
                correct_humans = sum(1 for value in propagated.values() if value == "+")
                h_t = correct_humans / total_humans if total_humans > 0 else 0  # Probability of human buzzing correctly

                # Model correctness and confidence
                g_t = 1 if run['correctness'] else 0  # Model correctness (1 if correct, 0 if wrong)
                c_t = np.exp(run['conf']) if elicit == 'logit' else run['conf']  # Model confidence (scaled if logit)

                # Compute necessary term
                hgc_sum += (1 - h_t) * g_t * c_t  # Expected probability adjusted for human buzzing
                count += 1
            

            # Compute expectation
            E_1_minus_h_g_c = hgc_sum / count if count > 0 else 0
            Grace_q_adjusted = 1 - E_1_minus_h_g_c  # Final computation

            results[model_key] = Grace_q_adjusted

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
                total_humans = len([team for team in team_per_question[question['tossup_index']]['seen'] if team.startswith('H')])# total number of humans who saw the question

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
    print('ece')
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
        print(model, len(conf_acc_model[model]))
    return ece_results

def calculate_ECE_bulk_by_run(packets, elicit, num_bins=10):
    print('ece')
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
        print(model, len(conf_acc_model[model]))
        ece_results[model] = calculate_ECE(conf_acc_model[model], num_bins)
        
    return ece_results


def calculate_brier_score(confidence_accuracy_pairs):
    
    brier_score_sum = 0
    for conf, acc in confidence_accuracy_pairs:
        brier_score_sum += (conf - acc) ** 2
    return brier_score_sum / len(confidence_accuracy_pairs)

def calculate_brier_score_bulk(packets, elicit):
    print('brier')
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
        print(model, len(conf_acc_model[model]))
        brier_results[model] = calculate_brier_score(conf_acc_model[model])
        
    return brier_results

def calculate_brier_score_bulk_by_run(packets, elicit):
    print('brier')
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
        print(model, len(conf_acc_model[model]))
        brier_results[model] = calculate_brier_score(conf_acc_model[model])
        
    return brier_results


def main():
    model_name_mapping = {
        "M1": "GPT-4",
        "M2": "GPT-4o",
        "M3": "Mistral-7b-Instruct",
        "M4": "LLama-2-70b-Chat",
        "M5": "Llama-3.1-8B-Instruct",
        "M6": "Llama-3.1-70B-Instruct"
    }

    ## logits
    combined_packets = []
    for packet in range(1, 13):
        packet_file = json.load(open(f'./model_human_output/packet{packet}_model_human_output.json', 'r'))
        combined_packets += packet_file

    ece_result = calculate_ECE_bulk(combined_packets, 'logit')
    brier_result = calculate_brier_score_bulk(combined_packets, 'logit')

    Grace_q_values_adjusted = [calculate_Grace_q_adjusted(row, 'logit') for row in combined_packets]
    Grace_q_result_adjusted = dict(pd.DataFrame(Grace_q_values_adjusted).mean(axis=0))

    Grace_q_values_non_adjusted = [calculate_Grace_q_non_adjusted(row, 'logit') for row in combined_packets]
    Grace_q_result_non_adjusted = dict(pd.DataFrame(Grace_q_values_non_adjusted).mean(axis=0))

    data = []
    metrics = ["Grace (human_non_adjusted)", "Grace (human_adjusted)", "ECE", "Brier Score"]

    for metric, values in zip(metrics, [Grace_q_result_non_adjusted, Grace_q_result_adjusted, ece_result, brier_result]):
        for model, value in values.items():
            mapped_model_name = model_name_mapping.get(model, model)  
            data.append({"Metric": metric, "Model": mapped_model_name, "Value": value})

    df = pd.DataFrame(data)
    for metric in metrics:
        ascending_order = metric != "SHq" 
        df.loc[df['Metric'] == metric, 'Rank'] = df[df['Metric'] == metric]['Value'].rank(ascending=ascending_order, method='dense').astype(int)
    df['Value'] = np.round(df['Value'], 3)

    df_with_rank = df.pivot(index='Model', columns='Metric', values=['Value', 'Rank'])
    formatted_df =  df_with_rank.apply(lambda x: x['Value'].astype(str) + " (" + x['Rank'].astype(int).astype(str) + ")", axis=1)
    formatted_df = formatted_df.reset_index()

    formatted_df.columns = ['Model', 'Brier Score', 'ECE', 'Grace (human_non_adjusted)', "Grace (human_adjusted)"]
    formatted_df = formatted_df.sort_values(by='Grace (human_adjusted)', key=lambda col: col.str.extract(r'\((\d+)\)')[0].astype(int))
    logit_based = formatted_df
    
    return logit_based


if __name__ == "__main__":
    main()
