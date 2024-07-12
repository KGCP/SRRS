import os
import rdflib
from SPARQLWrapper import SPARQLWrapper, Wrapper
import pandas as pd
import pprint
import regex as re
import numpy as np
import csv
import json

from tqdm import tqdm
import logging
import multiprocessing as mp
import time

from rule_processing_functions import *
from QueryExecutor import *

import pickle

#from sklearn.linear_model import LinearRegression       # linear regression
from sklearn.model_selection import train_test_split    # train-test split
#from sklearn.preprocessing import OneHotEncoder         # nominal variable
#from sklearn.preprocessing import Normalizer
#from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import MinMaxScaler
#from sklearn.preprocessing import OrdinalEncoder
from sklearn import metrics
#from sklearn.tree import DecisionTreeRegressor
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor

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
def load_attribute_rules(file_path, qe):
    with open(file_path, "r") as f:
        json_data = json.load(f)
    print(f'dictionary keys: {json_data.keys()}')
    att_rules = json_data['att_rules']
    print(f'No. of attribute rules loaded: {len(att_rules)}')

    return att_rules
        

def train_ph_model_dtr(dataset, features, target = 'pH_val_X'):
    #print("target :", target)
    #print("features :", features)
    #print(dataset[features].isna().sum())
    dataset = dataset[features].fillna(0)

    y = dataset[target]
    x = dataset.drop(target, axis=1)
    #x = x.drop(proxy, axis=1)
    #print("Data set size:", len(x))
    

    # train-test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

    dt_model = RandomForestRegressor(n_estimators=10) #DecisionTreeRegressor()
    dt_model.fit(x_train, y_train) #Trained using unscaled features and ordinally encoded soil texture
    #print("Train & Test Score with Decision Tree Regressor:", dt_model.score(x_train, y_train), dt_model.score(x_test, y_test))
    
    
    # Standard Metrics
    #metric_names = ['Training MSE', 'Testing MSE', 'Training RMSE', 'Testing RMSE', 'Training R-squared', 'Testing R-squared', 'Training MAE', 'Testing MAE']
    #dt_metrics = [metrics.mean_squared_error(y_train, dt_model.predict(x_train)), metrics.mean_squared_error(y_test, dt_model.predict(x_test)),
    #              np.sqrt(metrics.mean_squared_error(y_train, dt_model.predict(x_train))), np.sqrt(metrics.mean_squared_error(y_test, dt_model.predict(x_test))),
    #              metrics.r2_score(y_train, dt_model.predict(x_train)), metrics.r2_score(y_test, dt_model.predict(x_test)),
    #              metrics.mean_absolute_error(y_train, dt_model.predict(x_train)), metrics.mean_absolute_error(y_test, dt_model.predict(x_test))]

                  
    metrics_dict = {'Training MSE': metrics.mean_squared_error(y_train, dt_model.predict(x_train)), 
                    'Testing MSE': metrics.mean_squared_error(y_test, dt_model.predict(x_test)),
                    'Training RMSE': np.sqrt(metrics.mean_squared_error(y_train, dt_model.predict(x_train))), 
                    'Testing RMSE': np.sqrt(metrics.mean_squared_error(y_test, dt_model.predict(x_test))),
                    'Training R-squared': metrics.r2_score(y_train, dt_model.predict(x_train)), 
                    'Testing R-squared': metrics.r2_score(y_test, dt_model.predict(x_test)),
                    'Training MAE': metrics.mean_absolute_error(y_train, dt_model.predict(x_train)), 
                    'Testing MAE': metrics.mean_absolute_error(y_test, dt_model.predict(x_test))}
    #print(metrics_dict)
    #permutation importances
    #https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html
    importances_dt = permutation_importance(dt_model, x_train, y_train)

    #print(importances_dt)
    #Getting the Gini importance from the learnt Regression Tree
    # importance of a feature is computed as the (normalized) total reduction of the criterion brought by that feature.
    # It is also known as the Gini importance.
    # Warning: impurity-based feature importances can be misleading for high cardinality features (many unique values). 
    #https://scikit-learn.org/stable/modules/permutation_importance.html
    importances_df = pd.DataFrame({'feature name':x_train.columns, 
                                      'gini_imp': dt_model.feature_importances_,
                                       'permutation imp (mean)': np.round(importances_dt['importances_mean'],6),
                                       'permutation imp (std)': np.round(importances_dt['importances_std'],6)})
    #return metrics_df, pd.merge(left=coefficients_df, right=importances_df, on='feature name', how='left'), x, y
    
    #print(importances_df)
    return dt_model, metrics_dict, importances_df


def evaluate_attr_rule(qe, k, rule_string, summary_file_name, rule_metrics_file_name):  
    #first check support for the template
    rs, bs, hs, numerator_query, denominator_query, denominator_query_HC = compile_query_for_rule_v2(rule_string, optional_clause=True, verbrose=False)
    res_rs = qe.run_query_df(rs)
    rs = len(res_rs)
    
    #skipping if no rule support for the template
    if rs == 0:
        with open(summary_file_name, 'a') as f:
            f.write(f'\nrule template {k}: skipped')
            #global skipped_templates_count
            #skipped_templates_count += 1
        sys.stdout.flush()
        return (k, 0)
    else:
        #print(f'k: {k}, rs: {rs}')
        # ['d_X_from', 'd_X_to', 'd_pH_from', 'd_pH_to', 'm_X', 'm_pH', 'm_lime'] #not using these features
        features = ['pH_val_X', 'pH_val', 'C_val', 'Al_val', 'ECEC_val', 'lr_val', 'd_lime', 'culti_idx', 'd_culti_val', 'd_X_avg', 'd_pH_avg', 'gap' ]
        #'C_val', 'Al_val', 'ECEC_val'
        target = 'pH_val_X' #target pH value 
        available_features = [f for f in features if f in res_rs.columns]
        dt_model, metrics_dict, importances_df = train_ph_model_dtr(res_rs, available_features, target)
        tr_mse = metrics_dict['Training MSE']
        tst_mse = metrics_dict['Testing MSE']
        tr_rmse = metrics_dict['Training RMSE']
        tst_rmse = metrics_dict['Testing RMSE']
        tr_rsq = metrics_dict['Training R-squared']
        tst_rsq = metrics_dict['Testing R-squared']
        tr_mae = metrics_dict['Training MAE']
        tst_mae = metrics_dict['Testing MAE']
        
        features_string = importances_df['feature name'] #', '.join(importances_df['feature name']) 
        gini_importances = importances_df['gini_imp'] #
        permutation_imp_mean = importances_df['permutation imp (mean)'] #
        permutation_imp_std = importances_df['permutation imp (std)'] #
        with open(summary_file_name, 'a') as f:
            f.write(f'\nrule template {k}: rs: {rs}\tfeature importances:\n{importances_df}')
            
        with open(rule_metrics_file_name, 'a') as f:
            f.write(f'{k:<10} {rs:<10}'+
                    f' {np.round(tr_mse, 8):<10} {np.round(tst_mse, 8):<10}' +
                    f' {np.round(tr_rmse, 8):<10} {np.round(tst_rmse, 8):<10}' +
                    f' {np.round(tr_rsq, 8):<10} {np.round(tst_rsq, 8):<10}' +
                    f' {np.round(tr_mae, 8):<10} {np.round(tst_mae, 8):<10}' +
                    f' |\t{rule_string}\n')

        sys.stdout.flush()
        
        with open(output_folder+"/"+str(k)+"_dt_model"+".pickle", "wb") as output_file:
            pickle.dump(dt_model, output_file)
        
        return (k, 1)
    
# define callback function to collect the output in `results`
def collect_result(result):
    #global results
    #results.append(result)
    global results_count
    global no_results_count
    k, r = result
    if r==1:
        results_count += 1
    else:  #r==0
        no_results_count += 1
    #pbar_dict[k].update()
    pbar.update()

def collect_error_result(error):
    global errors_count
    errors_count +=1
    #global errors
    #errors.append(error)
    pbar.update()
    with open(summary_file_name, 'a') as f: 
        f.write(f'\n{error}')
    sys.stdout.flush()


    
if __name__ == '__main__':  
    
    # process hyper-parameters
    parameters = sys.argv[1:]
    if(len(parameters)) > 0:
        template_file, parallelize, from_rule, to_rule = parameters[0], bool(int(parameters[1])), int(parameters[2]), int(parameters[3]) # 0 0 5  # 1 0 -1
        #python RegressionTraining.py ./templates/lime_attribute_rules.json 0 0 2
    else:
        template_file = "./templates/lime_attribute_rules.json" #"./lime_rules_templates_new.json"
        parallelize = False   #True
        from_rule = 0
        to_rule = -1
        
    
    
    #First instantiate the QueryExecutor
    qe: QueryExecutor = QueryExecutor()
    
    rules = load_attribute_rules(template_file, qe)
    #rule_variants_dict = create_rule_variants(rules, combinations, comb_values)
    
    if to_rule == -1 or to_rule > len(rules):
        to_rule = len(rules) 
           
    log_file_name = './attr_rules/error_log_'+str(from_rule)+'_'+str(to_rule)+'.log'
    summary_file_name = './attr_rules/rule_summary_'+str(from_rule)+'_'+str(to_rule)+'.txt'
    rule_metrics_file_name = './attr_rules/regression_metrics_'+str(from_rule)+'_'+str(to_rule)+'.txt'
    output_folder = './trained_models/'
    
    
    #configure the logger
    logging.basicConfig(level=logging.ERROR, filename=log_file_name, filemode='w', 
                        format='%(asctime)s %(name)s - %(levelname)s - %(message)s', datefmt = '%Y-%m-%d %H:%M:%S', force=True)

    print(f'Evaluating attribute rules: {from_rule}-{to_rule} in {template_file}, Parallel processing: {parallelize}')
    with open(summary_file_name, 'w') as f:
        f.write(f'No. of attribute rules in {template_file}: {len(rules)}\n')
        f.write(f'Evaluating attribute rules: {from_rule}-{to_rule}, Parallel processing: {parallelize}')

    with open(rule_metrics_file_name, 'w') as f:
        f.write(f'''{"k":<10} {"RS":<10} {"Tr MSE":<10} {"Tst MSE":<10} {"Tr RMSE":<10} {"Tst RMSE":<10} {"Tr R-sq":<10} {"Tst R-sq":<10} {"Tr MAE":<10} {"Tst MAE":<10} |\tOriginal Rule\n''')

    T1 = time.time()

    # Step 1: Init multiprocessing.Pool()    
    if (parallelize):     
        num_proc = mp.cpu_count()//4 #mp.cpu_count()//2
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

        if (parallelize):
                #qe2: QueryExecutor = QueryExecutor()
                result = pool.apply_async(evaluate_attr_rule, args=(qe, k, rule_string, summary_file_name, rule_metrics_file_name), callback=collect_result, error_callback=collect_error_result)
                results_dict[k] = result

        else: # not parallel processing
                try:
                    result = evaluate_attr_rule(qe, k, rule_string, summary_file_name, rule_metrics_file_name)
                    results_dict[k] = result
                    k, r = result
                    if r==1:
                        results_count += 1
                    else:  #r==0
                        no_results_count += 1
                except Exception as e:
                    #print(f'rule {k}:{i} error: {e}')
                    logging.error(f'\nrule {k}:\n{rule_string}\n>>{e}', exc_info=(e))
                    errors_count +=1
                    with open(summary_file_name, 'a') as f: 
                        f.write(f'\nrule {k}:>>{e}')

                pbar.update()        
              


    if (parallelize):   
        # Step 3: Close Pool and let all the processes complete      
        pool.close()
        pool.join()  # postpones the execution of next line of code until all processes in the queue are done.

        #print error rules
        for k, res in results_dict.items():
            if not res.successful():
                try:
                    res.get()
                except Exception as e:
                    logging.error(f'\nrule {k}:\n{rules[k]}\n>>{e}', exc_info=(e))

    T2 = time.time()

    print(f'\n\nThe rule evaluation program runs about :{(T2 - T1)} s \nEvaluated {results_count}/{results_count+no_results_count} rules \nGot {errors_count} errors.')
    #print(f'\nSkipped {skipped_templates_count} rule templates.')
    #print(errors)

    with open(summary_file_name, 'a') as f:
        f.write(f'\n\nThe rule evaluation program runs about :{(T2 - T1)} s \nEvaluated {results_count}/{results_count+no_results_count} rules \nGot {errors_count} errors.')
        #f.write(f'\nSkipped {skipped_templates_count} rule templates.')


