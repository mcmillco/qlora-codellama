import pickle
from rapidfuzz import fuzz
import re
import networkx as nx
from tqdm import tqdm

dat = pickle.load(open("predictions/callgraph_java_codellama.pkl", "rb"))
print(dat[5].keys())

def fuzz_path_similarity(path1, path2):
    similarity = fuzz.ratio(path1, path2) / 100  # Convert to 0-1 scale
    return similarity
'''
total_score = 0
# path levenshtein score
for info in dat:
    true_paths = info["path"]
    gpt_paths = info["gpt_path"]
    totla_method_score = 0

    gpt_paths = re.findall(r'path:\s*(.*)', gpt_paths)
    for true_path in true_paths:
        true_path = [p.replace("\"", "") for p in true_path]
        true_path = '->'.join(true_path)
        total_temp_score = 0
        for gpt_path in gpt_paths:
            similarity = fuzz_path_similarity(true_path, gpt_path)
            total_temp_score += similarity
        if(gpt_paths != []):
            total_temp_score = total_temp_score / len(gpt_paths) 
    total_score += total_temp_score 
mean_levenshtein_score  = total_score / len(dat)
print(f"levenshtein score: {mean_levenshtein_score}")
'''
#  graph edit distance 
#total_distance = 0

#for info in tqdm(dat[:]):
#    true_paths = info["true_path"]
#    true_callgraph = [[item.strip('"') for item in sublist] for sublist in true_paths]
#    gpt_paths = info["gpt_path"]
#    gpt_paths = re.findall(r'path:\s*(.*)', gpt_paths)
#    gpt_callgraph = []
#    for path in gpt_paths:
#        path = path.split("->")
#        gpt_callgraph.append(path)
#    gpt_callgraph_edges = []
#    true_callgraph_edges = []
#    for sequence in gpt_callgraph:
#        edges = [(sequence[i], sequence[i + 1]) for i in range(len(sequence) - 1)]
#        gpt_callgraph_edges.extend(edges)
    
#    for sequence in true_callgraph:
#        edges = [(sequence[i], sequence[i + 1]) for i in range(len(sequence) - 1)]
#        true_callgraph_edges.extend(edges)
    
#    truegraph = nx.DiGraph()
#    truegraph.add_edges_from(true_callgraph_edges)


#    gptgraph = nx.DiGraph()
#    gptgraph.add_edges_from(gpt_callgraph_edges)


#    distance = nx.graph_edit_distance(truegraph, gptgraph)
#    total_distance += distance

#mean_edit_distance =  total_distance / len(dat)
#print(f"mean edit distance: {mean_edit_distance}")

# jaccarc similarity 
total_similarity = 0

for info in tqdm(dat[:]):
    new_true_paths = []
    true_paths = info["path"]
    for edges in true_paths:
        new_edges = [edge.split(".")[-1] for edge in edges] 
        new_true_paths.append(new_edges)
    true_callgraph = [[item.strip('"') for item in sublist] for sublist in new_true_paths]
    llm_results = info["result"]
    gpt_paths = llm_results.split("\n")
    #gpt_paths = re.findall(r'path:\s*(.*)', llm_results)
    gpt_callgraph = []
    for path in gpt_paths:
        path = path.split("->")
        gpt_callgraph.append(path)
    gpt_callgraph_edges = []
    true_callgraph_edges = []
    for sequence in gpt_callgraph:
        edges = [(sequence[i], sequence[i + 1]) for i in range(len(sequence) - 1)]
        gpt_callgraph_edges.extend(edges)
    
    for sequence in true_callgraph:
        edges = [(sequence[i], sequence[i + 1]) for i in range(len(sequence) - 1)]
        true_callgraph_edges.extend(edges)
    
    

    truegraph = nx.DiGraph()
    truegraph.add_edges_from(true_callgraph_edges)


    gptgraph = nx.DiGraph()
    gptgraph.add_edges_from(gpt_callgraph_edges)

    
    edges_true = set(truegraph.edges())
    edges_gpt = set(gptgraph.edges())
    
    intersection = len(edges_true & edges_gpt)
    union = len(edges_true | edges_gpt)
    jaccard_similarity = intersection / union if union != 0 else 0
    
    total_similarity += jaccard_similarity
    
mean_jaccard_similarity =  total_similarity / len(dat)
print(f"mean jaccard similarity: {mean_jaccard_similarity}")

# pair accuracy 

total_acc = 0

for info in tqdm(dat[:]):
    true_paths = info["path"]
    true_callgraph = [[item.strip('"') for item in sublist] for sublist in true_paths]

    llm_results = info["result"]
    gpt_paths = llm_results.split("\n")
    gpt_callgraph = []
    for path in gpt_paths:
        path = path.split("->")
        gpt_callgraph.append(path)
    gpt_callgraph_edges = []
    true_callgraph_edges = []
    for sequence in gpt_callgraph:
        edges = [(sequence[i], sequence[i + 1]) for i in range(len(sequence) - 1)]
        gpt_callgraph_edges.extend(edges)
    
    for sequence in true_callgraph:
        edges = [(sequence[i], sequence[i + 1]) for i in range(len(sequence) - 1)]
        true_callgraph_edges.extend(edges)
    
    truegraph = nx.DiGraph()
    truegraph.add_edges_from(true_callgraph_edges)


    gptgraph = nx.DiGraph()
    gptgraph.add_edges_from(gpt_callgraph_edges)

    
    edges_true = list(truegraph.edges())
    edges_gpt = list(gptgraph.edges())
    intersect_edges = list(set(edges_gpt).intersection(edges_true))
    number_of_intersect_edges = len(intersect_edges)
    if(edges_gpt != []):
        total_acc += number_of_intersect_edges / len(edges_gpt)
    elif(edges_gpt == [] and edges_true ==[]):
        total_acc += 1

mean_pair_accuracy = total_acc / len(dat)
print(f"mean pair accuracy: {mean_pair_accuracy}")




total_chain_acc = 0


for info in dat[:]:
    #print("------", info, "\n")
    true_edges = info["path"]
    gpt_paths = info["result"]
    totla_method_score = 0
    
    true_paths = []

    gpt_paths = re.findall(r'path:\s*(.*)', gpt_paths)
    for edge in true_edges:
        edge = [e.replace("\"", "") for e in edge]
        true_paths.append('->'.join(edge).strip())
    #print( true_paths, "\n")
    for gpt_path in gpt_paths:
        if(gpt_path.strip() in true_paths):
            total_chain_acc += 1
    #print(gpt_paths, "\n")
    if(len(gpt_paths) != 0):
        total_chain_acc /= len(gpt_paths)
    elif(gpt_paths == [] and true_paths == []):
        total_chain_acc += 1
mean_chain_accuracy = total_chain_acc / len(dat)
print(f"mean chain accuracy: {mean_chain_accuracy}")



