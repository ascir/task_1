from model_implementation import get_narrative, root_folder

query_file = f'{root_folder}Queries.txt'
narr = get_narrative(query_file)
print(narr)