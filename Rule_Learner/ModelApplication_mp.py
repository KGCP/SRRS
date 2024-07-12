# coding=UTF-8
import pandas as pd
import regex as re
import numpy as np
import json
import os
import logging
import multiprocessing as mp
import time

from rule_processing_functions import *
from TripleStore import *
from QueryExecutor import *  #Use the GraphDB endpoint instead of rdflib

import pickle
#from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import builtins
builtins.MEL_config_filename = "config-FutureSOILS" # default configuration file; "Only-UseCase" param. is required!
import MEL
from MEL import FutureSOILS_PPP_Engine  # This import should be declared after the previous line to avoid loading TNNT.


import warnings 
# Settings the warnings to be ignored 
warnings.filterwarnings('ignore') 

#function borrowed from Sergio's code FutureSOILS_PPP_Engine.py
def P_normaliseName(name):
    if (name is None): return None  # None
    normalised = name\
        .replace('/'  ,"_Pr_")\
        .replace("µ"  ,"_Mi_")\
        .replace('%'  ,"_PC_")\
        .replace('('  ,"_LP_")\
        .replace(')'  ,"_RP_")\
        .replace('{'  ,"_LB_")\
        .replace('}'  ,"_RB_")\
        .replace("+"  ,"_Pl_")\
        .replace("*"  ,"_As_")\
        .replace("|"  ,"_VB_")
    normalised = re.sub('[^A-Za-z0-9_\-\.]+', '-', normalised)  # "Crop Analysis" -> "Crop-Analysis"; "1:5 (CaCl2)" -> "1-5-(CaCl2)";
    return re.sub('\.$', "__Dt", normalised)  # if ends with a '.' -> "__Dt"

def get_question_details():
    q = f'''SELECT ?X_obs ?X_res ?label ?la ?type ?moa ?uom
    WHERE {{ 
    ?A fs-onto:hasObservation ?X_obs .
    ?X_obs sosa:observedProperty ?type .
    ?X_obs sosa:usedProcedure ?moa .
    ?X_obs sosa:hasResult ?X_res .
    ?X_res rdf:type fs-onto:Question .
    ?X_res rdfs:label ?label .
    #BIND (str(?label_str) AS ?label) .
    OPTIONAL {{?X_res qudt:unit ?uom}} .
    }}
    '''
    result =  data.result_to_df(data.run_query(q))
    
    return result

    
def construct_shortcut_name(obs_type, moa, uom):
    obs_type_short = obs_type.replace('fs-data:ObservedProperty-', '')
    moa_short  = moa.replace('fs-data:MoA-', '')
    #uom_short  = '_' if uom==None else '_'+uom.replace('fs-data:UoM-', '')+'_'
    uom_short  = '' if row.uom==None else '_'+row.uom.replace('fs-data:UoM-', '')+'_'
    if LOCAL_processing:
        #normalised_str = P_normaliseName(f'{obs_type_short}__({moa_short}){uom_short}')
        normalised_str = P_normaliseName(f'{obs_type_short}__({moa_short})_{uom_short}')
    else: #running on SERVER
        #normalised_str = FutureSOILS_PPP_Engine.KG.P_normaliseName(f'{obs_type_short}__({moa_short}){uom_short}')
        normalised_str = FutureSOILS_PPP_Engine.KG.P_normaliseName(f'{obs_type_short}__({moa_short})_{uom_short}')
    return 'fs-data:'+normalised_str+'_PSC', 'fs-data:'+normalised_str+'_ASC'

def construct_shortcut_name_df(row):
    obs_type_short = row.type.replace('fs-data:ObservedProperty-', '')
    moa_short  = row.moa.replace('fs-data:MoA-', '')
    #uom_short  = '_' if row.uom==None else '_'+row.uom.replace('fs-data:UoM-', '')+'_'
    uom_short  = '' if row.uom==None else '_'+row.uom.replace('fs-data:UoM-', '')+'_'
    if LOCAL_processing:
        #normalised_str = P_normaliseName(f'{obs_type_short}__({moa_short}){uom_short}')
        normalised_str = P_normaliseName(f'{obs_type_short}__({moa_short})_{uom_short}')
    else: #running on SERVER
        normalised_str = FutureSOILS_PPP_Engine.KG.P_normaliseName(f'{obs_type_short}__({moa_short}){uom_short}')
        normalised_str = FutureSOILS_PPP_Engine.KG.P_normaliseName(f'{obs_type_short}__({moa_short})_{uom_short}')
    return 'fs-data:'+normalised_str+'_PSC', 'fs-data:'+normalised_str+'_ASC'


def generate_testcases(path_in, file_name, ASC=False):
    questions = get_question_details()

    if len(questions) > 0:
        questions[['PSC_name', 'ASC_name']] = questions.apply(construct_shortcut_name_df, axis=1, result_type='expand')
    
    with open(path_in+file_name+'.txt', "w") as f:
        for i, row in questions.iterrows():
            if row.moa != None:
                q = row.label.split('^^')[0].strip('"')
                #f.write(f'fs-data:pH___LP_1-5-CaCl2_RP___PSC({row.X_obs}, ?{q})\n')
                if ASC == True:
                    f.write(f'{row.ASC_name}({row.X_obs}, ?{q}), sosa:hasResult({row.X_obs}, {row.X_res})\n')
                else: #PSC
                    f.write(f'{row.PSC_name}({row.X_obs}, ?{q}), sosa:hasResult({row.X_obs}, {row.X_res})\n')
            else:
                print(f'Skipping unsupported prediction: {row.label}, {row.type}, {row.moa}, {row.uom}')
    print(f'{len(questions)} testcases were written to {path_in}{file_name}.txt')

    
def load_template_descriptions(path_in, file_name):
    template_descriptions_dict = dict()
    #template_descriptions_fname = "./templates/lime_rules_templates_descriptions.txt"
    with open(path_in+file_name+'.txt', "r") as f:
        template_desc = f.readlines()
    for line in template_desc:
        template_descriptions_dict[int(line.split('\t')[0].strip('i:').strip(','))] = line.strip()
    return template_descriptions_dict


def load_testcases(path_in, file_names):
    all_test_cases = []
    for file_name in file_names:
        with open(path_in+file_name+'.txt', "r") as f:
            test_cases = f.readlines()
        all_test_cases.extend(test_cases)
    #print(f'Loaded {len(all_test_cases)} test_cases from {len(file_names)} files.')
    logging.info(f'Loaded {len(all_test_cases)} test_cases from {len(file_names)} files.')
    return all_test_cases

def find_regression_metrics(path_in):
    file_names = os.listdir(path_in)
    file_names_valid = []
    for fname in file_names:
        if fname.startswith("regression_metrics"):
            file_names_valid.append(fname)
    return file_names_valid

def load_attribute_rules(path_in, file_names):
    rules_dict = dict()
    #all_rules = []
    for file_name in file_names:
        with open(path_in+file_name, "r")  as f: #"./output/rule_metrics - Copy.txt"
            rules = f.readlines()
            
            for rule_line in rules[1:]:
                metrics_str, select_exp, bind_exp, filter_exp, rule_exp = rule_line.split('\t')
                # BS: Body Suuport, RS: Rule Support, HS: Head Support, SC: Standard Confidence, HC: Head Coverage
                k, rs, tr_mse, tst_mse, tr_rmse, tst_rmse, tr_rsq, tst_rsq, tr_mae, tst_mae = metrics_str.split()[:-1]
                bind_exp = bind_exp[5:-1] if ((bind_exp.startswith("BIND[") and bind_exp.endswith("]")) and len(bind_exp) > 6) else None
                filter_exp = filter_exp[7:-1] if ((filter_exp.startswith("FILTER[") and filter_exp.endswith("]")) and len(filter_exp) > 8) else None
                select_exp = select_exp[7:-1] if ((select_exp.startswith("SELECT[") and select_exp.endswith("]")) and len(select_exp) > 8) else None
                #print(k, i, BS, RS, HS, SC, HC)

                rule = re.sub(r'\t|\n|<=|\(|\)|,', ' ', rule_exp).split() 
                head = np.array(rule[:3])
                #print(f'>>head: {head}')
                bodys = np.array(rule[3:]) #np.array(rule[3:]).reshape(int(len(rule[3:]) / 3), 3)
                #print(f'>>bodys: {bodys}')

                is_connected_closed = connected_closed_rule_check(rule)  

                is_X_in_head = True if ('?X' in head and '?Y' not in head) else False

                rule_metrics = {'RuleSupport':int(rs), 
                                'TrainingMSE':float(tr_mse), 'TestingMSE':float(tst_mse),
                                'TrainingRMSE':float(tr_rmse), 'TestingRMSE':float(tst_rmse),
                                'TrainingR_sq':float(tr_rsq), 'TestingR_sq':float(tst_rsq),
                                'TrainingMAE':float(tr_mae), 'TestingMAE':float(tst_mae),
                                'model':int(k)}
                rule_type = {'is_connected_closed':is_connected_closed }#, 'is_ac2':is_ac2, 'is_ac1':is_ac1, 'index_head':index_head, 'is_X_in_head':is_X_in_head}
                #template_details = template_descriptions_dict[int(k)]
                #print(rule_head, rule_bodies)
                key =  tuple(head)
                #rules_dict[key] = rules_dict.get(key, []) + [(rule_metrics, select_exp, bind_exp, filter_exp, head, bodys, rule_type, rule_line, template_details)]
                rules_dict[key] = rules_dict.get(key, []) + [{'rule_metrics':rule_metrics, 'select_exp':select_exp, 'bind_exp':bind_exp, 'filter_exp':filter_exp, 'head':head, 'bodys':bodys, 'rule_type':rule_type, 'rule_string':rule_line}]
        #all_rules.extend(rules[1:])
    #print(f'Loaded {len(all_rules)} rules from {len(file_names)} files.')
    
    #print(len(rules_dict))
    
    #for k, v in rules_dict.items():
    #    print(f'k:{k}, {len(v)}')
    return rules_dict


def load_regression_models(path_in):
    models_dict = dict()
    file_names = os.listdir(path_in)
    
    for fname in file_names:
        if fname.endswith("_dt_model.pickle"):
            try:
                k = int(fname.split('_')[0])
                with open(path_in+fname, 'rb') as handle:
                    models_dict[k] = pickle.load(handle)
            except Exception as e:
                logging.error(f'\n>>{e}', exc_info=(e))
                print(f'Error loading pickle file {fname}>> {e}')

    
    return models_dict


def construct_query_line(body):
    query_line = body[1] + ' ' + body[0] + ' ' + body[2] + ' .\n'
    return query_line

def get_entity_counts_in_rule(rule_string):
    select_exp, bind_exp, filter_exp, rule_exp = rule_string.split('\t')
    
    #split the rule by the following characters (<= will be the head, body separator)
    #ignores the optional blocks { } and filter not exists blocks (~{  })
    rule = re.sub(r'\t|\n|<=|\(|\)|\{|\}|,|\~', ' ', rule_exp).split()
       
    head = np.array(rule[:3])
    #print(f'>>head: {head}')
    bodys = np.array(rule[3:]).reshape(int(len(rule[3:]) / 3), 3)
    
    # counting the entities in the rule body only
    i = 0
    entity_counts = dict()
    #entity_counts[head[1]] = 1
    #entity_counts[head[2]] = 1
    while i < len(bodys):
        entity_counts[bodys[i][1]] = entity_counts.get(bodys[i][1], 0) + 1
        entity_counts[bodys[i][2]] = entity_counts.get(bodys[i][2], 0) + 1
        i += 1
        
    return entity_counts

def apply_attribute_rule(data, rule, s, p, o, true_val, bound, bound_loc):
    #rule_metrics, select_exp, bind_exp, filter_exp, head, bodys, rule_type = rule
    #print(bodys)
    
    #rule_body_reshaped = np.array(bodys)#.reshape(int(len(bodys) / 3), 3)

    head = rule['head']
    bodys= rule['bodys']
    bound_variable = head[bound_loc]
    to_replace = s if bound_loc==1 else o    #replace subject or object?
    head_bound = np.where(head==bound_variable, to_replace, head) #replace the bound variable in rule read
    rule_body = np.where(bodys==bound_variable, to_replace, bodys) #replace the bound variable in rule body
    ans_loc = 2 if bound_loc==1 else 1
    
    
    rule_string = rule['rule_string'].split('|')[1:][0][1:]
    #rs, bs, hs, numerator_query, denominator_query, denominator_query_HC = compile_query_for_rule_v2(rule_string, optional_clause=True, verbrose=False)
    #print("to be replaced >>", bound_variable, to_replace)
    #print(bs)
    #res_bs = data.result_to_df(data.run_query(bs))
    #bs = len(res_bs)    
    ## only get the required observation (row)
    #results = res_bs[res_bs[bound_variable.lstrip('?')]==to_replace]
        
    entity_counts = get_entity_counts_in_rule(rule_string)
    
    bind_filter_string = ""
    if rule['bind_exp'] is not None: #BIND[]
        clauses = rule['bind_exp'].split(',')
        for k in range(len(clauses)):
            bind_filter_string += '\nBIND ' + clauses[k] + ' .'
    if rule['filter_exp'] is not None: #FILTER[]
        clauses = rule['filter_exp'].split(',')
        for k in range(len(clauses)):
            bind_filter_string += '\nFILTER ' + clauses[k] + ' .'
    
  
    query_string = query_prefix
    query_string += 'SELECT'
    for k in entity_counts.keys():
        if is_general_entity(k) and k != bound_variable:
            #Using the Stored Procedure (DB.DBA.QName) defined by Sergio at the VOS endpoint
            # to make the proper replacement of the namespace IRI to the pre-defined prefix.
            #query_string += ' sql:QName(' + k + ')' # ' ' + k
            query_string += ' ' + k
            
    #add the specific expressions to the SELECT clause
    if rule['select_exp'] is not None: #SELECT[]
        clauses = rule['select_exp'].split(',')
        for k in range(len(clauses)):
            query_string += ' ' + clauses[k]
    #for body in bodys_bound:
    #    query_string = query_string + construct_query_line(body)  
    
    query_string += f'''
    FROM {named_graph}
    FROM <http://soco.cecc.anu.edu.au/onto/FutureSOILS#>
    WHERE {{ '''
    
    j=0
    while(j<len(rule_body)):
        query_string += '\n'
        if rule_body[j].startswith('{~{'): #start of optional block with immediate filter not exists (NO SPACE in '{~{')
            query_string += "OPTIONAL {\nFILTER NOT EXISTS {\n"
            query_string += rule_body[j+1] + ' ' + rule_body[j].lstrip('{~{') + ' ' + rule_body[j+2] + ' .'  #strip '{~{' if found
        elif rule_body[j].startswith('{'): #start of an optional block
            query_string += "OPTIONAL {\n"
            query_string += rule_body[j+1] + ' ' + rule_body[j].lstrip('{') + ' ' + rule_body[j+2] + ' .'  #strip '{' if found
        elif rule_body[j].startswith('~{'): #start of an filter not exists block
            query_string += "FILTER NOT EXISTS {\n"
            query_string += rule_body[j+1] + ' ' + rule_body[j].lstrip('~{') + ' ' + rule_body[j+2] + ' .'  #strip '~{' if found
        else: 
            query_string += rule_body[j+1] + ' ' + rule_body[j] + ' ' + rule_body[j+2] + ' .'
        if (j+3 < len(rule_body)) and rule_body[j+3].startswith('}'): #closing an optional block
            query_string += "\t}}" if rule_body[j+3]=='}}' else  "\t}"
            j += 4 #skip the curly braces and move to next triple
        else:
            j+= 3 #move to next triple

    query_string += bind_filter_string + '\n}'
    #print(query_string)
    
    #results = data.result_to_df(data.run_query(query_string))
    results = data.run_query_df(query_string)
    
    #print(results)
    if len(results) > 0:
        return len(results), results 
    else:
        return 0, None

    
def apply_ph_model_dtr(k, dataset, features, target = 'pH_val_X'):
    x = dataset[features]
    #for col in x.columns:
    #    x[col] = x[col].apply(lambda x: x.split("^^")[0].strip('"') if x != None else None).astype(float)
        
    x = x.fillna(0)
    y = models_dict[k].predict(x)
    return y


def apply_model_to_testcase(data, k, rule, s, p, o, true_val, bound, bound_loc, ans_loc, template_descriptions_dict):
    if data==None:
        data: QueryExecutor = QueryExecutor(prefix_file_path=base_kg_file_path, sparql_endpoint=endpoint)#'http://localhost:7200/sparql')
        
    len_results, res = apply_attribute_rule(data, rule, s, p, o, true_val, bound, bound_loc)         
    #print("attribute rule #", rule['rule_metrics']['template'], len_results)
    #print(res.columns)
                    

    if res is not None:
        features = ['pH_val_X', 'pH_val', 'C_val', 'Al_val', 'ECEC_val', 'lr_val', 'd_lime', \
                    'culti_idx', 'd_culti_val', 'd_X_avg', 'd_pH_avg', 'gap' ]
        #'C_val', 'Al_val', 'ECEC_val'
        target = 'pH_val_X' #target pH value 
        available_features = [f for f in features if f in res.columns]
        y = apply_ph_model_dtr(rule['rule_metrics']['model'], res, available_features, target = 'pH_val_X')
        #print("y = ", y)
                        
        #assert len_results==1 
        metrics = rule['rule_metrics'].copy()
        metrics['model_details'] = template_descriptions_dict[metrics['model']] #rule[-1]
        metrics['rule'] = rule['rule_string'] #rule[-2]
        metrics['model_type'] = "RandomForestRegressor" #"DecisionTreeRegressor" 
        #ans_list.append(metrics)
                        
        #print(rule['rule_metrics'])
        rule_metrics = rule['rule_metrics']
        logging.info(f'prediction({o}, {k[ans_loc]}): {y}, {rule_metrics}')
                        

        return k, p, s, o, y, metrics       
    else:
        return k, p, s, o, [], [] 
                        
    
# define callback function to collect the output in `results`
def collect_result(result):
    #global results_dict
    global finished_count
    #k, p, s, o, y, metrics = result
    finished_count += 1
    #print(f'{finished_count}', end=', ')
    #if y is not None: 
    #    for y_val in y:
    #        results_dict[f'{p}({s}, {o})'][y_val] = results_dict.get(f'{p}({s}, {o})', dict()).get(y_val, []) + [metrics]
    #print(f'{finished_count}')
        
def collect_error_result(error):
    print(f'\n{error}')
    sys.stdout.flush()
    
if __name__ == '__main__': 
    # process hyper-parameters
    parameters = sys.argv[1:]
    if(len(parameters)) > 0:
        ttl_file_name = parameters[0]
        named_graph = f'<http://soco.cecc.anu.edu.au/data/FutureSOILS/{parameters[1]}/>'
    else:
        ttl_file_name = '(2024-05-22) - Other - TestingMPGDB (Test).xlsx - FS_KG.ttl'
        named_graph = f'<http://soco.cecc.anu.edu.au/data/FutureSOILS/{parameters[1]}/>'
        

    query_prefix = '''
PREFIX fs-onto: <http://soco.cecc.anu.edu.au/onto/FutureSOILS#>
PREFIX fs-data: <http://soco.cecc.anu.edu.au/data/FutureSOILS/>
PREFIX sosa: <http://www.w3.org/ns/sosa/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
'''
    
    SERVER = 'http://localhost:7200'  # you may want to change this
    REPOSITORY = 'FutureSOILS'               # you most likely want to change this
    endpoint = f'{SERVER}/repositories/{REPOSITORY}'  # standard path for GraphDB queries
    
    #Load the ontology and the user input data into the TripleStore
    rdf_format= 'ttl'
    
    #path_prefix = './'
    path_prefix = '/data/projects/FutureSOILS/code/2-PM/Rule_Learner/' 
    local_path_out = path_prefix + 'predictions/'
    templates_path = path_prefix + 'templates/'
    
    rules_path = path_prefix + 'rules/'
    logs_path = path_prefix + '_logs/'
    
    attr_rules_path = path_prefix + '/attr_rules/'
    trained_models_path = path_prefix + 'trained_models/'

    
    LOCAL_processing = False #Set to True if running locally, False if running on Server 
    
    if LOCAL_processing:
        #onto_file_path = "/data/projects/FutureSOILS/onto/FutureSOILS-ontology.ttl" 
        #path_in = "/data/projects/FutureSOILS/data/KG/" 
        #path_out = "/data/projects/FutureSOILS/data/JSON/"
        #base_kg_file_path: "/data/projects/FutureSOILS/onto/FutureSOILS-base-kg.ttl"
        
        onto_file_path = "./FutureSOILS-ontology_29_02_2024.ttl"   #location of the ontology file
        path_in = "./test_KG/"   #location of the loaded ttl files
        path_out = "./predictions/" #where to save the predictions
        base_kg_file_path: "./FutureSOILS-base-kg.ttl"
        
        # The suffix for the JSON files:
        json_suffix_binned_predictions = '_binned_predictions.json'
        json_suffix_value_predictions = '_value_predictions.json'
        
        # The suffix for the log files:
        logs_suffix_binned_predictions = '_binned_predictions.log'
        logs_suffix_value_predictions = '_value_predictions.log'
    
    else:
        C_ENV: str = MEL.Utils.DATASET["$env"]  # environment config? server or local
        PARAMS: dict = MEL.Utils.DATASET["RDF-Mappings"]
        onto_file_path: str = PARAMS["File"][f"{C_ENV}Ontology"]
        path_in: str = PARAMS["Output"][f"{C_ENV}Folder"]
        path_out: str = MEL.Utils.DATASET["RESTful-API"][f"{C_ENV}Rule-Application-JSON-Folder"]

        # The suffix for the JSON files:
        json_suffix_binned_predictions = MEL.Utils.DATASET["RESTful-API"]["Rule-Application-Bins-JSON-Suffix"]
        json_suffix_value_predictions = MEL.Utils.DATASET["RESTful-API"]["Rule-Application-Values-JSON-Suffix"]
        
        # The suffix for the log files:
        logs_suffix_binned_predictions = MEL.Utils.DATASET["RESTful-API"]["Rule-Application-Bins-LOG-Suffix"]
        logs_suffix_value_predictions = MEL.Utils.DATASET["RESTful-API"]["Rule-Application-Values-LOG-Suffix"]
        
        base_kg_file_path: str = PARAMS["File"][f"{C_ENV}KG-Base"]

    #print()
    
    print(f'named_graph: {named_graph}')
    print(f'base_kg_file_path: {base_kg_file_path}')
    print(f'endpoint: {endpoint}')
        
    #configure the logger
    log_file_name = f'{logs_path}{ttl_file_name}{logs_suffix_value_predictions}'
    logging.basicConfig(level=logging.INFO, filename=log_file_name, filemode='w', 
                        format='%(asctime)s %(name)s - %(levelname)s - %(message)s', datefmt = '%Y-%m-%d %H:%M:%S', force=True)

    T1 = time.time()
    data: TripleStore = TripleStore(onto_file_path, path_in, [ttl_file_name], rdf_format)   
    #qe: QueryExecutor = QueryExecutor()
    
    #generate testcases from the question entities and save in a file
    generate_testcases(local_path_out, f'{ttl_file_name}_testcases_ASC', ASC=True) #generate ASC testcases
    test_cases = load_testcases(local_path_out, [f'{ttl_file_name}_testcases_ASC'])

    
    #load rule template descriptions
    template_descriptions_dict = load_template_descriptions(templates_path, 'lime_attribute_rules_descriptions')
    
    #load the rules to be applied
    #rules_dict = load_rules(rules_path, ['rule_metrics_0_32'])#, 'rule_metrics_32_64'])# 'rule_metrics_32_64', 'rule_metrics_64_96', 'rule_metrics_128_160'])
    rules_dict = load_attribute_rules(attr_rules_path, find_regression_metrics(attr_rules_path))

    models_dict = load_regression_models(trained_models_path)
    print(f'Loaded {len(models_dict)} trained regression models')
    logging.info(f'Loaded {len(models_dict)} trained regression models')
    
    # Step 1: Init multiprocessing.Pool()       
    num_proc = max(min(mp.cpu_count()//2, len(test_cases)*len(models_dict)),1)
    print(f'using {num_proc} processes')
    pool = mp.Pool(num_proc)
        
    # # Step 2: Use loop to parallelize 
    #Apply the rules over the testcases
    
    #Apply the rules over the testcases
    results_dict = dict()
    mp_results_dict = dict()
    result_entity_mapping = dict()  #To collect the Result entity information
    finished_count = 0

    for test_case in test_cases:
        #print('=======================================\n\ntest case: ', test_case.strip())
        logging.info(f'=======================================\ntest case: {test_case.strip()}')
        split_testcase = re.sub(r'\t|\n|<=|\(|\)|\{|\}|,', ' ', test_case).split() #each line is of format p(s, o)
        p, s, o = split_testcase[0:3]
        if len(split_testcase) == 6:
            #print(p, s, o, split_testcase[3:])
            p1, s1, res_entity = split_testcase[3:]
            if p1=="sosa:hasResult" and s1==s:
                result_entity_mapping[s1] = res_entity
        #print(s, p, o)
        variable = o if o.startswith('?') else s if s.startswith('?') else None
        if variable == None:
            continue #no question is being asked, skip to the next testcase
        else: #assume the other one is bound
            bound = s if o.startswith('?') else o 
            bound_loc = 1 if o.startswith('?') else 2
        #print(f'predicate: {p}, variable: {variable}, bound: {bound}, bound_loc: {bound_loc}' )
        results_dict[f'{p}({s}, {o})'] = dict()
        true_val = None

        for k,v in rules_dict.items():
            #print(k[0], p)
            if k[0] == p:                        
                #print(">>>> matched key:", k)
                is_X_in_head = '?X' in k
                is_Y_in_head = '?Y' in k
                if (is_X_in_head and bound_loc==1) or (is_Y_in_head and bound_loc==2):
                    #print(f'\n>>>> matched key: {k}, No. of rules to check: {len(v)}') 
                    logging.info(f'>>>> question: {variable}, matched key: {k}, No. of models to check: {len(v)}') 
                        
              
                else:
                    continue
                    
                ans_loc = 2 if bound_loc==1 else 1
                #ans_list = []
                for rule in v:
                    result = pool.apply_async(apply_model_to_testcase, args=(None, k, rule, s, p, o, true_val, bound, bound_loc, ans_loc, template_descriptions_dict), callback=collect_result, error_callback=collect_error_result)
                    #mp_results_dict[f'{variable}, {k}, {v}, {rule}'] = result
                    rule_str = rule['rule_string']
                    mp_results_dict[f'{k}, {p}, {s}, {o}, {len(v)}, {rule_str}'] = result

                    
    print('closing pool')                
    # Step 3: Close Pool and let all the processes complete      
    pool.close()
    pool.join()  # postpones the execution of next line of code until all processes in the queue are done.
    print(f'completed {finished_count} processes')
    #print error rules
    for k, res in mp_results_dict.items():
        if not res.successful():
            try:
                res.get()
                
            except Exception as e:
                logging.error(f'\nerror {k}:\n>>{e}', exc_info=(e))
        else:
            k, p, s, o, y, metrics = res.get()
            if y is not None: 
                for y_val in y:
                    results_dict[f'{p}({s}, {o})'][y_val] = results_dict.get(f'{p}({s}, {o})', dict()).get(y_val, []) + [metrics]

    T2 = time.time()
    print(f'\n\nThe model application program runs about :{(T2 - T1)} s \nWriting results into json files...')
    logging.info(f'The model application program runs about :{(T2 - T1)}s. Writing results into json files...')
    
    #convert the results into a format that will contain all the information needed to convert into the FS KG representation                 
    results_list = []
    for k, v in results_dict.items():
        #print(k)
        p, s, o = re.sub(r'\t|\n|<=|\(|\)|\{|\}|,', ' ', k).split()
        if len(v) > 0:
            for ans_key, answers in v.items():
                #print(ans_key)
                results_list.append(
                {"question": k, 
                 "from_entity": s,
                 "question_entity": result_entity_mapping.get(s),
                 "property": p,
                 "predicted_value": ans_key,
                 "predicted_value_datatype": "xsd:float",
                 "matched_models": answers
                })
        else:
            #print(None)
            results_list.append(
            {"question": k, 
            "from_entity": s,
            "question_entity": result_entity_mapping.get(s),
            "property": p,
            "predicted_value": None,
            "predicted_value_datatype": "xsd:float",
            "matched_models": []
            }) 
                    
    #save the binned predictions (simplified format)
    with open(f'{local_path_out}{ttl_file_name}_simple{json_suffix_value_predictions}', "w") as f:
        json.dump({'value_predictions':results_dict}, f) 
       
    #save the binned predictions (KG processing format)
    with open(f'{path_out}{ttl_file_name}{json_suffix_value_predictions}', "w") as f:
        json.dump({'value_predictions':results_list}, f) 
        
    logging.info(f'Prediction outputs were written to {path_out}{ttl_file_name}{json_suffix_value_predictions}')
    print(f'Prediction outputs were written to {path_out}{ttl_file_name}{json_suffix_value_predictions}')
    
    