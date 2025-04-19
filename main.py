from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
import os
from equivalence_checker import EquivalenceChecker
import yaml

def create_adjacency_list(filename, stop=20):
    """
    This function creates an adjacency list from the implications.txt file.
    The file is expected to have the following format:
    #0 (897x) ∃x0 (¬∃y1 (Pet(y1) ∧ Owned(claire,y1,2:00) ∧ Fed(claire,y1,x0)) ∧ ∀x1 ((Pet(x1) ∧ Owned(max,x1,2:00)) → Fed(max,x1,x0)))
    
    Also implements transitive closure - if x is connected to y, and y is connected to z, then x is connected to z.
    """
    adj_list = defaultdict(list)
    formulas = dict()
    f = open(filename, "r")
    i=0
    prev_tab = 0
    prev_formula = ""
    for line in f:
        if i == stop:
            break
        i += 1
        line = line.rstrip()
        line=list(line)
        tabs = 0
        while line[0:4] == [' ', ' ', ' ', ' ']:
            line = line[4:]
            tabs += 1
        if line[0:4] == ['*', '*', '*', ' ']:
            line = line[4:]
        line = ''.join(line)
        line = line.split(" ")
        formula_id = line[0]
        formula = " ".join(line[2:])
        formulas[formula_id] = formula
        if tabs > prev_tab:
            adj_list[prev_formula].append(formula_id)
        prev_tab = tabs
        prev_formula = formula_id
    
    # Implement transitive closure
    # We'll use the Floyd-Warshall algorithm to compute all reachable nodes
    all_nodes = set(formulas.keys()).union(set(node for nodes in adj_list.values() for node in nodes))
    
    # Create a copy of the adjacency list to work with
    transitive_adj_list = {node: set(adj_list[node]) if node in adj_list else set() for node in all_nodes}
    
    # Apply the transitive closure
    for k in all_nodes:
        for i in all_nodes:
            if i in transitive_adj_list and k in transitive_adj_list[i]:  # If i is connected to k
                for j in list(transitive_adj_list[k]):  # For all nodes j connected to k
                    transitive_adj_list[i].add(j)  # Add an edge from i to j
    
    # Convert sets back to lists for the final adjacency list
    final_adj_list = {node: list(targets) for node, targets in transitive_adj_list.items()}
    
    return final_adj_list, formulas

def visualize_graph(output_file, adj_list, formulas, max_nodes=100, title="Formula Implications Graph"):
    """
    Visualize the graph represented by the adjacency list.
    
    Parameters:
    - output_file: File to save the graph image to
    - adj_list: Dictionary where keys are formula IDs and values are lists of IDs that the key implies
    - formulas: Dictionary mapping formula IDs to their text representation
    - max_nodes: Maximum number of nodes to display (to avoid overcrowding)
    - title: Title of the graph
    """
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes and edges
    nodes_added = 0
    for source, targets in adj_list.items():
        if nodes_added >= max_nodes:
            break
        if source not in G:
            G.add_node(source)
            nodes_added += 1
        
        for target in targets:
            if target not in G:
                if nodes_added >= max_nodes:
                    break
                G.add_node(target)
                nodes_added += 1
            G.add_edge(source, target)
    
    # Set up the plot
    plt.figure(figsize=(12, 10))
    
    # Use a layout that works well for directed graphs
    pos = nx.spring_layout(G, seed=42)  # for reproducible layout
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue', alpha=0.8)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, edge_color='gray', 
                          arrowstyle='-|>', arrowsize=15)
    
    # Draw labels with truncated formula text
    labels = {}
    for node in G.nodes():
        if node in formulas:
            # Truncate formula text to avoid overcrowding
            formula_text = formulas[node]
            if len(formula_text) > 20:
                formula_text = formula_text[:17] + "..."
            labels[node] = f"{node}"
        else:
            labels[node] = node
    
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_family='sans-serif')
    
    # Add title and remove axes
    plt.title(title)
    plt.axis('off')
    
    # Add statistics
    plt.figtext(0.05, 0.02, f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    
    # Show the plot
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Graph image saved to {output_file}")
    
    # Return the graph object in case further analysis is needed
    return G

def write_adjacency_list_to_file(adj_list, output_folder, filename="adjacency_list.txt"):
    """
    Writes the adjacency list to a file.
    
    Parameters:
    - adj_list: Dictionary where keys are formula IDs and values are lists of IDs that the key implies
    - output_folder: Folder to write the file to
    - filename: Name of the file to write to
    """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Create full path
    filepath = os.path.join(output_folder, filename)
    
    with open(filepath, "w") as f:
        for source, targets in adj_list.items():
            f.write(f"{source}: {','.join(targets)}\n")
    print(f"Adjacency list written to {filepath}")
    return filepath

def remove_keys_from_adjacency_list(adj_list, keys_to_remove):
    """
    Removes a set of keys from the adjacency list.
    
    Parameters:
    - adj_list: Dictionary where keys are formula IDs and values are lists of IDs that the key implies
    - keys_to_remove: Set or list of keys to remove from the adjacency list
    
    Returns:
    - A new adjacency list with the specified keys removed (both as keys and as values in target lists)
    """
    # Create a new adjacency list
    new_adj_list = defaultdict(list)
    
    # Copy the adjacency list excluding the keys to remove
    for source, targets in adj_list.items():
        if source not in keys_to_remove:
            # Filter out targets that are in keys_to_remove
            new_targets = [target for target in targets if target not in keys_to_remove]
            if new_targets:  # Only add if there are targets left
                new_adj_list[source] = new_targets
    
    return new_adj_list

def find_highest_degree_nodes(adj_list):
    """
    Finds all nodes with the highest degree in the adjacency list.
    
    Parameters:
    - adj_list: Dictionary where keys are formula IDs and values are lists of IDs that the key implies
    
    Returns:
    - A dictionary with information about the highest degree nodes
    """
    # Calculate the degree of each node (out-degree + in-degree)
    degrees = defaultdict(int)
    
    # Count out-degrees (nodes that this node points to)
    for source, targets in adj_list.items():
        degrees[source] += len(targets)
    
    # Count in-degrees (nodes that point to this node)
    for source, targets in adj_list.items():
        for target in targets:
            degrees[target] += 1
    
    # Find the highest degree
    if not degrees:
        return {
            "highest_degree_nodes": [],
            "degree": 0,
            "details": {}
        }
    
    max_degree = max(degrees.values())
    
    # Find all nodes with the highest degree
    highest_degree_nodes = [node for node, degree in degrees.items() if degree == max_degree]
    
    return {
        "highest_degree_nodes": highest_degree_nodes,
        "degree": max_degree,
        "details": {node: degrees[node] for node in highest_degree_nodes}
    }

def insufficient_formula(formula, lexicon):
    """
    Checks if a formula is invalid by determining if it uses all terms in the lexicon.
    Returns True if the formula is invalid.
    
    Parameters:
    - formula: The formula text to check
    - lexicon: List of terms that should be present in the formula
    """
    for word in lexicon:
        if word not in formula:
            return True
    return False

def insufficient_formula_list(formulas, lexicon):
    """
    Identifies invalid formulas that don't use the right lexicon.
    
    Parameters:
    - formulas: Dictionary of formula_id -> formula_text
    - lexicon: List of terms that should be present in valid formulas
    
    Returns:
    - List of formula IDs that are invalid
    """
    invalid_keys = []
    for formula_id, formula in formulas.items():
        if insufficient_formula(formula, lexicon):
            invalid_keys.append(formula_id)
    print(f"Invalid formulas: {invalid_keys}")
    return invalid_keys

def equivalence_checker(formulas, valid_answers):
    """
    Checks if formulas are equivalent to any of the valid answers.

    Parameters:
    - formulas: Dictionary of formula_id -> formula_text
    - valid_answers: List of valid answer formulas
    
    Returns:
    - List of formula IDs that are not equivalent to any valid answer
    """
    # Convert valid answers to the notation used by EquivalenceChecker
    standardized_answers = []
    for answer in valid_answers:
        standardized = answer.replace("∧", "&").replace("∨", "|").replace("¬", "~").replace("∀", "A").replace("∃", "E")
        standardized_answers.append(standardized)
    
    invalid_keys = []
    for formula_id, formula in formulas.items():
        formula_standardized = formula.replace("∧", "&").replace("∨", "|").replace("¬", "~").replace("∀", "A").replace("∃", "E")
        
        # Check if the formula is equivalent to any valid answer
        is_equivalent = False
        for answer in standardized_answers:
            if EquivalenceChecker().check_equivalence(formula_standardized, answer):
                is_equivalent = True
                break
        
        if not is_equivalent:
            invalid_keys.append(formula_id)
    
    return invalid_keys

def load_yaml_to_variables(file_path):
    """
    Load a YAML file and assign each top-level key-value pair to a variable with the same name.
    Returns a dictionary of the loaded variables.
    """
    try:
        # Open and read the YAML file
        with open(file_path, 'r', encoding='utf-8') as file:
            yaml_data = yaml.safe_load(file)
        
        # Create variables in the global scope with the same names as the keys
        variables = {}
        for key, value in yaml_data.items():
            globals()[key] = value
            variables[key] = value
            
        print(f"Successfully loaded variables: {', '.join(variables.keys())}")
        return variables
    
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

# Main workflow
def main():

    # Load configuration from YAML file
    config_file = "config.yml"
    config = load_yaml_to_variables(config_file)
    if config is None:
        print("Failed to load configuration. Exiting.")
        return


    lexicon = config.get("lexicon", [])
    implication_file = config.get("implication_file", "implication.txt")
    output_folder = config.get("output_folder", "output")
    valid_answers = config.get("valid_answers", [])
    
    incorrect_by_question_ambiguity = ["#6", "#378", "#23", "#345", "#3"]
    
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Load the initial adjacency list and formulas
    print("Creating adjacency list...")
    adj_list, formulas = create_adjacency_list(implication_file, 50000)
    
    # Identify and remove invalid nodes
    print("Identifying invalid formulas...")
    invalid_keys = insufficient_formula_list(formulas, lexicon)
    print(f"Found {len(invalid_keys)} invalid formulas out of {len(formulas)} total formulas")
    
    # Add known incorrect formulas
    invalid_keys += incorrect_by_question_ambiguity
    print(f"Invalid keys after adding incorrect by question ambiguity: {invalid_keys}")
    
    # Check equivalence with valid answers
    print("Checking formula equivalence with valid answers...")
    non_equivalent_keys = equivalence_checker(formulas, valid_answers)
    invalid_keys += non_equivalent_keys
    print(f"Invalid keys after equivalence check: {len(invalid_keys)}")
    
    # Remove the invalid nodes from the adjacency list
    print("Removing invalid nodes from adjacency list...")
    filtered_adj_list = remove_keys_from_adjacency_list(adj_list, invalid_keys)
    print(f"Filtered adjacency list has {len(filtered_adj_list)} nodes and {sum(len(targets) for targets in filtered_adj_list.values())} edges")
    
    # Identify and remove nodes with no edges
    print("Removing nodes with no edges...")
    empty_nodes = [node for node, targets in filtered_adj_list.items() if not targets]  
    for node in empty_nodes:
        del filtered_adj_list[node]
    print(f"Removed {len(empty_nodes)} nodes with no edges")
    
    # Get filtered graph statistics
    filtered_nodes = len(filtered_adj_list)
    filtered_edges = sum(len(targets) for targets in filtered_adj_list.values())
    print(f"Filtered graph has {filtered_nodes} nodes and {filtered_edges} edges")
    # Export the filtered graph image
    print("Visualizing filtered graph...")
    filtered_graph_path = os.path.join(output_folder, "formula_implications_graph_filtered.png")
    G_filtered = visualize_graph(filtered_graph_path, filtered_adj_list, formulas,
                                max_nodes=5000, title="Filtered Formula Implications Graph")
    
    # Find highest degree nodes in the filtered graph
    highest_degree_info_filtered = find_highest_degree_nodes(filtered_adj_list)
    print(f"Highest degree nodes in filtered graph: {highest_degree_info_filtered['highest_degree_nodes']}")
    print(f"Degree: {highest_degree_info_filtered['degree']}")
    
    # Save both adjacency lists to files
    write_adjacency_list_to_file(adj_list, output_folder, "adjacency_list_initial.txt")
    write_adjacency_list_to_file(filtered_adj_list, output_folder, "adjacency_list_filtered.txt")
    
    print("Process completed successfully")

if __name__ == "__main__":
    main()
