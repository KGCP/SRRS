import os
import rdflib
from SPARQLWrapper import SPARQLWrapper, Wrapper
import pandas as pd
import pprint
import regex as re
import numpy as np
import csv
import json
from itertools import product
from functools import reduce

from tqdm import tqdm
import logging
import multiprocessing as mp
import time

from rule_processing_functions import *
from QueryExecutor import *

# IMPORTANT: Requires secure VPN connection to ANU (Global Protect VPN)
#   And Access to the VOS endpoint if running locally
#   Using SSH tunnelling and port forwarding
# For example, the below command will allow you to access the service running on port 8890 on dirt.gpu by pointing at port 8890 on localhost .
#   user@local-desktop:~$ ssh -L 8890:localhost:8890 u1234567@dirt.gpu -J u1234567@bulwark.cecs.anu.edu.au

# Or configure the .ssh/config file and use command `ssh dirt_sparql`
#   the contents of ~/.ssh/id_rsa.pub should have been already added to testgpu-01:/home/admin/.ssh/authorized_keys , 
#   as well as bulwark:/home/users/u1234567/.ssh/authorized_keys .

'''
 Host dirt_sparql
 HostName dirt.gpu
 ProxyJump u1234567@bulwark.cecs.anu.edu.au
 User u4975316
 ForwardX11 yes
 IdentityFile ~/.ssh/id_rsa
 LocalForward 8890 localhost:8890
''' 



##############################################################    
def load_rule_templates(file_path, qe):
    with open(file_path, "r") as f:
        json_data = json.load(f)
    print(f'dictionary keys: {json_data.keys()}')
    rules = json_data['rules']
    combinations = json_data['combinations']
    print(f'No. of rule templates loaded: {len(rules)}')

    comb_values = dict()
    for key, val in combinations.items():
        if key=="skos:broader":
            q = f'''SELECT DISTINCT ?o  WHERE {{ ?s {key} ?o . ?s rdf:type fs-onto:CultivationTreatment .}}''' 
        else:
            q = f'''SELECT DISTINCT ?o  WHERE {{ ?s {key} ?o . }}'''

        results = qe.run_query(q)
        comb_values[key] = [qe.get_qname(res['o']['value']) for res in results['results']['bindings']]
    print(f'combinations loaded: {comb_values.keys()}')

    #sort the values
    for k, v in comb_values.items():
        v.sort()
        
    return rules, combinations, comb_values
        
def create_rule_variants(rule_string, combinations, comb_values):
    select_exp, bind_exp, filter_exp, rule_exp = rule_string.split('\t') #rules[i].split('\t')[0], rules[i].split('\t')[1], rules[i].split('\t')[2], rules[i].split('\t')[3]
    rule = re.sub(r'\t|\n|<=|\(|\)|\{|\}|,|\~', ' ', rule_exp).split()     
    #rule = re.sub(r'\t|\n|<=|\(|\)|,', ' ', rule_exp).split()

    head = np.array(rule[:3])
    bodys = np.array(rule[3:]).reshape(int(len(rule[3:]) / 3), 3)
    items = np.insert(bodys, 0, head, axis=0) # combine both head and body elements
    #print(f'>>items: {items[:,0]}') # all the predicates 
    
    variables_to_replace = dict()
    for item in items:
        if item[0] in combinations.keys():
            #print(item[0], combinations[item[0]])
            #if item[1] in combinations[item[0]]: 
            #    variables_to_replace[item[1]] = comb_values[item[0]]
            if item[2] in combinations[item[0]]: #assuming that the replacement will be in the object (o) in <p(s,o)> pattern in rule
                variables_to_replace[item[2]] = comb_values[item[0]]

    #gerenate rule variants by replacing the variables with all possible literal combinations
    rule_variants_list = []
    for p in product(*[product(v) for v in variables_to_replace.values()]): #generate all possible replacement combinations 
        replacements = [(alias,alias_replace) for alias, alias_replace in  zip(variables_to_replace.keys(), [r[0] for r in p])]
        #print(replacements)
        rule_variants_list.append(reduce(lambda a, kv: a.replace(*kv), replacements, rule_string))

    return rule_variants_list

# define callback function to collect the output in `results`
def collect_result(result):
    #global results
    #results.append(result)
    global results_count
    global no_results_count
    k, i, r = result
    if r==1:
        results_count += 1
    else:  #r==0
        no_results_count += 1
    #pbar_dict[k].update()
    #pbar.update()

def collect_error_result(error):
    global errors_count
    errors_count +=1
    #global errors
    #errors.append(error)
    #pbar.update()
    with open(summary_file_name, 'a') as f: 
        f.write(f'\n{error}')
    sys.stdout.flush()


    
if __name__ == '__main__':  
    
    # process hyper-parameters
    parameters = sys.argv[1:]
    if(len(parameters)) > 0:
        template_file, parallelize, from_rule, to_rule = parameters[0], bool(int(parameters[1])), int(parameters[2]), int(parameters[3]) # 0 0 5  # 1 0 -1
        #python RuleEvaluation.py ./templates/lime_rules_templates.json 1 176 192
    else:
        template_file = "./templates/lime_rules_templates.json" #"./lime_rules_templates_new.json"
        parallelize = False   #True
        from_rule = 0
        to_rule = -1
        
    
    
    #First instantiate the QueryExecutor
    qe: QueryExecutor = QueryExecutor()
    
    rules, combinations, comb_values = load_rule_templates(template_file, qe)
    #rule_variants_dict = create_rule_variants(rules, combinations, comb_values)
    
    if to_rule == -1 or to_rule > len(rules):
        to_rule = len(rules) 
           
    log_file_name = './output/error_log_'+str(from_rule)+'_'+str(to_rule)+'.log'
    summary_file_name = './output/rule_summary_'+str(from_rule)+'_'+str(to_rule)+'.txt'
    rule_metrics_file_name = './output/rule_metrics_'+str(from_rule)+'_'+str(to_rule)+'.txt'
    
    
    #configure the logger
    logging.basicConfig(level=logging.ERROR, filename=log_file_name, filemode='w', 
                        format='%(asctime)s %(name)s - %(levelname)s - %(message)s', datefmt = '%Y-%m-%d %H:%M:%S', force=True)

    print(f'Evaluating templates: {from_rule}-{to_rule} in {template_file}, Parallel processing: {parallelize}')
    with open(summary_file_name, 'w') as f:
        f.write(f'No. of rule templates in {template_file}: {len(rules)}\n')
        f.write(f'Evaluating templates: {from_rule}-{to_rule}, Parallel processing: {parallelize}')

    with open(rule_metrics_file_name, 'w') as f:
        f.write(f'''{"k":<10} {"i":<10} {"BS":<10} {"RS":<10} {"HS":<10} {"SC=RS/BS":<10} {"HC=RS/HS":<10} |\tOriginal Rule\n''')

    T1 = time.time()

    # Step 1: Init multiprocessing.Pool()    
    if (parallelize):     
        num_proc = mp.cpu_count()//2
        print(f'using {num_proc} processes')
        pool = mp.Pool(num_proc)

    # # Step 2: Use loop to parallelize    
    errors_count = 0
    #errors = []
    #results = []
    results_dict = dict() 
    #pbar_dict = dict() #to store the multiple pbars 
    results_count = 0
    no_results_count = 0
    skipped_templates_count = 0
    pbar = tqdm(total = to_rule-from_rule)

    #for k, v in rule_variants_dict.items(): #iterate through the rule templates
    #for k, rule_string in enumerate(rules):
    for k in range(from_rule, to_rule):
        rule_string = rules[k]
        results_dict[k] = dict()
        #if k!=2:
        #    continue
        #print(f'k = {k}')
        
        #first check support for the template
        rs, bs, hs, numerator_query, denominator_query, denominator_query_HC = compile_query_for_rule_v2(rule_string, optional_clause=True, verbrose=False)
        denominator = int(qe.run_query_df(denominator_query).iloc[0]['count'])
        
        #skipping if no body support for the template
        if denominator == 0:
            with open(summary_file_name, 'a') as f:
                f.write(f'\nrule template {k}: skipped')
                pbar.update()
                skipped_templates_count += 1
            continue
        
        rule_variants = create_rule_variants(rule_string, combinations, comb_values)
        
        with open(summary_file_name, 'a') as f:
            f.write(f'\nrule template {k}: {len(rule_variants)} variants, bs: {denominator}')
            
 
        #pbar_dict[k] = tqdm(total = len(v)) #, position=k)
        #pbar =  tqdm(total = len(v)) #, position=k)
        #print(k, len(v))
        for i, rule in enumerate(rule_variants): #iterare through the rule variants of a template
                       
            # creating the SPARQL qureies with optional_clause=True 
            if (parallelize):
                #qe2: QueryExecutor = QueryExecutor()
                result = pool.apply_async(evaluate_rules, args=(None, k, i, rule, True, rule_metrics_file_name), callback=collect_result, error_callback=collect_error_result)
                results_dict[k][i] = result

            else: # not parallel processing
                try:
                    result = evaluate_rules(qe, k, i, rule, True, rule_metrics_file_name)
                    results_dict[k][i] = result
                    k, i, r = result
                    if r==1:
                        results_count += 1
                    else:  #r==0
                        no_results_count += 1
                except Exception as e:
                    #print(f'rule {k}:{i} error: {e}')
                    logging.error(f'\nrule {k}:{i}\n{rule}\n>>{e}', exc_info=(e))
                    errors_count +=1
                    with open(summary_file_name, 'a') as f: 
                        f.write(f'\nrule {k}:{i} >>{e}')
                #pbar.update() #pbar_dict[k].update()
        pbar.update()
            #if i>=10:
            #    break

    if (parallelize):   
        # Step 3: Close Pool and let all the processes complete      
        pool.close()
        pool.join()  # postpones the execution of next line of code until all processes in the queue are done.

        #print error rules
        for k, res_dict in results_dict.items():
            for i, res in res_dict.items(): #some will be empty due to skipping the template
                if not res.successful():
                    try:
                        res.get()
                    except Exception as e:
                        logging.error(f'\nrule {k}:{i}\n{rule_variants_dict[k][i]}\n>>{e}', exc_info=(e))

    T2 = time.time()

    print(f'\n\nThe rule evaluation program runs about :{(T2 - T1)} s \nEvaluated {results_count}/{results_count+no_results_count} rules \nGot {errors_count} errors.')
    print(f'\nSkipped {skipped_templates_count} rule templates.')
    #print(errors)

    with open(summary_file_name, 'a') as f:
        f.write(f'\n\nThe rule evaluation program runs about :{(T2 - T1)} s \nEvaluated {results_count}/{results_count+no_results_count} rules \nGot {errors_count} errors.')
        f.write(f'\nSkipped {skipped_templates_count} rule templates.')


