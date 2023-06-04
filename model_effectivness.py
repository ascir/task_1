import glob
import os
from math import log2


def precision12(relevance_folder, model_results, model_regex, output_file):
    # Dictionary to store the relevance judgments
    relevance_data = {}
    precision_12 = []
    # Read the relevance judgments
    for file_name in os.listdir(relevance_folder):
        file_path = os.path.join(relevance_folder, file_name)
        if os.path.isfile(file_path) and file_path.endswith(".txt"):
            with open(file_path, 'r') as file:
                content = file.read().splitlines()
                for line in content:
                    query_id, doc_id, relevance = line.split()
                    query_id = query_id.strip()
                    doc_id = doc_id.strip()
                    relevance = int(relevance.strip())
                    if query_id not in relevance_data:
                        relevance_data[query_id] = {}
                    relevance_data[query_id][doc_id] = relevance
    # Iterate over the score files
    with open(f'{output_file}', 'w') as map_file:
        for file_path in glob.glob(os.path.join(model_results, model_regex)):
            query_id = os.path.basename(file_path).split("_")[1].split("Ranking")[0]
        
            with open(file_path, 'r') as file:
                content = file.read().splitlines()
        
                relevant_docs = 0
        
                for i, line in enumerate(content):
                    if i >= 12:
                        break
                    doc_id, score = line.split()
                    doc_id = doc_id.strip()
                    score = float(score.strip())
        
                    relevance = relevance_data.get(query_id, {}).get(doc_id, 0)
        
                    if relevance == 1:
                        relevant_docs += 1

                precision = relevant_docs/12
                precision_12.append(precision)
                map_file.write(f"{query_id} {precision}\n")
            
    map_file.close()
                
    return precision_12

     
def avg_precision(relevance_folder, model_results, model_regex, output_file):
    # Dictionary to store the relevance judgments
    relevance_data = {}
    sum_precision = 0
    # Read the relevance judgments
    for file_name in os.listdir(relevance_folder):
        file_path = os.path.join(relevance_folder, file_name)
        if os.path.isfile(file_path) and file_path.endswith(".txt"):
            with open(file_path, 'r') as file:
                content = file.read().splitlines()
                for line in content:
                    query_id, doc_id, relevance = line.split()
                    query_id = query_id.strip()
                    doc_id = doc_id.strip()
                    relevance = int(relevance.strip())
                    if query_id not in relevance_data:
                        relevance_data[query_id] = {}
                    relevance_data[query_id][doc_id] = relevance
    average_precision_list = []
    # Iterate over the score files
    with open(f'{output_file}', 'w') as map_file:
        for file_path in glob.glob(os.path.join(model_results, model_regex)):
            query_id = os.path.basename(file_path).split("_")[1].split("Ranking")[0]
        
            with open(file_path, 'r') as file:
                content = file.read().splitlines()
        
                relevant_docs = 0
                total_precision = 0.0
                query_precision = []
        
                for i, line in enumerate(content):
                    doc_id, score = line.split()
                    doc_id = doc_id.strip()
                    score = float(score.strip())
        
                    relevance = relevance_data.get(query_id, {}).get(doc_id, 0)
        
                    if relevance == 1:
                        relevant_docs += 1
                        precision = relevant_docs / (i + 1)
                        query_precision.append(precision)
                        total_precision += precision
        
                if relevant_docs > 0:
                    average_precision = total_precision / relevant_docs
                    sum_precision += average_precision
                    average_precision_list.append(average_precision)
                    map_file.write(f"{query_id} {average_precision}\n")
        mean_average_precision = sum_precision/len(average_precision_list)
        map_file.write(f"Mean Average precision: {mean_average_precision}\n")
    map_file.close()
                
    return average_precision_list


def dcg(relevance_folder, model_results, model_regex, output_file):
    # Dictionary to store the relevance judgments
    relevance_data = {}
    dcg_precision = []
    
    # Read the relevance judgments
    for file_name in os.listdir(relevance_folder):
        file_path = os.path.join(relevance_folder, file_name)
        if os.path.isfile(file_path) and file_path.endswith(".txt"):
            with open(file_path, 'r') as file:
                content = file.read().splitlines()
                for line in content:
                    query_id, doc_id, relevance = line.split()
                    query_id = query_id.strip()
                    doc_id = doc_id.strip()
                    relevance = int(relevance.strip())
                    if query_id not in relevance_data:
                        relevance_data[query_id] = {}
                    relevance_data[query_id][doc_id] = relevance

    # Iterate over the score files
    with open(f'{output_file}', 'w') as map_file:
        for file_path in glob.glob(os.path.join(model_results, model_regex)):
            query_id = os.path.basename(file_path).split("_")[1].split("Ranking")[0]
        
            with open(file_path, 'r') as file:
                content = file.read().splitlines()
        
                cumulative_gain = 0.0
        
                for i, line in enumerate(content):
                    if i >= 12:
                        break

                    doc_id, score = line.split()
                    doc_id = doc_id.strip()
                    score = float(score.strip())

                    relevance = relevance_data.get(query_id, {}).get(doc_id, 0)
                    gain = relevance / log2(i + 2)  # Calculate gain for each rank position

                    cumulative_gain += gain
                map_file.write(f"{query_id} {cumulative_gain}\n")
                dcg_precision.append(cumulative_gain)

    map_file.close()
                
    return dcg_precision