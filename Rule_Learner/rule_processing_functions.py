#import os
#import rdflib
#from SPARQLWrapper import SPARQLWrapper, Wrapper
#import pandas as pd
#import pprint
import regex as re
import numpy as np
#import csv
#import json
from itertools import product
from functools import reduce
import sys

from QueryExecutor import *

###################################################################
# Rule checks
###################################################################

# extract rules from rule-based system
def rule_extract(filepath):
    rules = []
    with open(filepath, "r") as f:
        data = f.readlines()
        rules = [rule for rule in data]
    return rules

    
# check if the rule is closed and connected rule
def connected_closed_rule_check(rule):
    rule_head = rule[:3]
    return rule_head[1] == '?X' and rule_head[2] == '?Y'


# check if the entity is a general entity in the form of "?X" for example
# general entities (variables) start with the ? character
def is_general_entity(entity):
    #return len(entity) == 1
    return entity.startswith('?')




###################################################################
# Converting a rule string to SPARQL Query
###################################################################
# Reads a rule in AnyBURL format and compiles a SPARQL query to retrieve the matching body patterns from the graph
def compile_query_for_rule(rule_string, optional_clause=True, verbrose=True):
    #print(f'\n{rules[i]}')
    #assume these will be tab separated
    select_exp, bind_exp, filter_exp, rule_exp = rule_string.split('\t') #rules[i].split('\t')[0], rules[i].split('\t')[1], rules[i].split('\t')[2], rules[i].split('\t')[3]

    #print("select :", select_exp, len(select_exp))
    #print("bind :", bind_exp, len(bind_exp))
    #print("filter :", filter_exp, len(filter_exp))
    
    #split the rule by the following characters (<= will be the head, body separator)
    #ignores the optional blocks { }
    # assming that the first triple can't be optional and that last triple can't be optional in case of connected closed rules
    rule = re.sub(r'\t|\n|<=|\(|\)|\{|\}|,', ' ', rule_exp).split()
       
    head = np.array(rule[:3])
    #print(f'>>head: {head}')
    bodys = np.array(rule[3:]).reshape(int(len(rule[3:]) / 3), 3)
    #print(f'>>bodys: {bodys}')
    
    # counting the entities in the rule body only
    i = 0
    entity_counts = dict()
    #entity_counts[head[1]] = 1
    #entity_counts[head[2]] = 1
    while i < len(bodys):
        entity_counts[bodys[i][1]] = entity_counts.get(bodys[i][1], 0) + 1
        entity_counts[bodys[i][2]] = entity_counts.get(bodys[i][2], 0) + 1
        i += 1
        
    
    bind_filter_string = ""
    if (bind_exp.startswith("BIND[") and bind_exp.endswith("]")) and len(bind_exp) > 6: #BIND[]
        clauses = str(bind_exp[5:-1]).split(',')
        for k in range(len(clauses)):
            bind_filter_string += '\nBIND ' + clauses[k] + ' .'
    if (filter_exp.startswith("FILTER[") and filter_exp.endswith("]")) and len(filter_exp) > 8: #FILTER[]
        clauses = str(filter_exp[7:-1]).split(',')
        for k in range(len(clauses)):
            bind_filter_string += '\nFILTER ' + clauses[k] + ' .'
            
    query_string = 'SELECT'
    for k in entity_counts.keys():
        if is_general_entity(k):
            #Using the Stored Procedure (DB.DBA.QName) defined by Sergio at the VOS endpoint
            # to make the proper replacement of the namespace IRI to the pre-defined prefix.
            #query_string += ' sql:QName(' + k + ')' # ' ' + k
            query_string += ' ' + k
            
    #add the specific expressions to the SELECT clause
    if (select_exp.startswith("SELECT[") and select_exp.endswith("]")) and len(select_exp) > 8: #SELECT[]
        clauses = str(select_exp[7:-1]).split(',')
        for k in range(len(clauses)):
            query_string += ' ' + clauses[k]
    
    # Splitting rule again, removed the substitution of curly braces which denote optional blocks
    if optional_clause:
        rule_body = re.sub(r'\t|\n|<=|\(|\)|,', ' ', rule_exp).split()[3:]
    else:
        rule_body = re.sub(r'\t|\n|<=|\(|\)|\{|\}|,', ' ', rule_exp).split()[3:] #also remove the curly braces
    #print(rule_body)
    query_string += '\nWHERE { ' 
    
    # supports upto two nested OPTIONAL blocks {.....{...}...} or {.....{...}}
    # assming that the first triple can't be optional
    j=0
    while(j<len(rule_body)):
        query_string += '\n'
        if rule_body[j].startswith('{'): #start of an optional block
            query_string += "OPTIONAL {\n"
        query_string += rule_body[j+1] + ' ' + rule_body[j].lstrip('{') + ' ' + rule_body[j+2] + ' .'  #strip '{' if found
        if (j+3 < len(rule_body)) and rule_body[j+3].startswith('}'): #closing an optional block
            query_string += "\t}}" if rule_body[j+3]=='}}' else  "\t}"
            j += 4 #skip the curly braces and move to next triple
        else:
            j+= 3 #move to next triple

    query_string += bind_filter_string + '\n}'

    if verbrose:
        print("\nQUERY FOR RULE BODY MATCHES:")
        print(query_string)
    
    head_string = 'SELECT'
    if is_general_entity(head[1]):
        head_string += ' ' + head[1]
    if is_general_entity(head[2]):
        head_string += ' ' + head[2]
    head_string += '\nWHERE { '
    head_string += '\n' + head[1] + ' ' + head[0] + ' '  + head[2] + ' .'
    head_string += '\n}'
    
    if verbrose:
        print("\nQUERY FOR RULE HEAD MATCHES:")
        print(head_string) 
    
    #construct the whole rule match query (including both rule body and rule head)
    head_body_string = query_string[0:-1] + head_string.split('\n')[-2] + '\n}'
    if verbrose:
        print("\nQUERY FOR RULE BODY AND HEAD MATCHES:")
        print(head_body_string) 
    
    if '?X' in entity_counts.keys() and '?Y' in entity_counts.keys():
        distinct_count_string = 'SELECT (COUNT(*) AS ?count) {' + '\n' + 'SELECT DISTINCT ?X ?Y' + '\n'
    else: #ASSUME either X or Y to be present
        distinct_count_string = 'SELECT (COUNT (DISTINCT ' + ('?X' if '?X' in entity_counts.keys() else '')  + ('?Y' if '?Y' in entity_counts.keys() else '') + ') AS ?count)' + '\n'
    numerator_string = distinct_count_string + ('\n').join(head_body_string.split('\n')[1:]) 
    denominator_string = distinct_count_string + ('\n').join(query_string.split('\n')[1:]) 
    denominator_string_HC = distinct_count_string + ('\n').join(head_string.split('\n')[1:])
    if '?X' in entity_counts.keys() and '?Y' in entity_counts.keys():
        numerator_string += '\n}'
        denominator_string += '\n}'
        denominator_string_HC += '\n}'
        
    if verbrose:
        print("\nQUERY FOR NUMERATOR:")
        print(numerator_string)
    
        print("\nQUERY FOR DENOMINATOR:")
        print(denominator_string)
    
        print("\nQUERY FOR DENOMINATOR OF HEAD COVERAGE:")
        print(denominator_string_HC)
    
    return head_body_string, query_string, head_string, numerator_string, denominator_string, denominator_string_HC

# Reads a rule in AnyBURL format and compiles a SPARQL query to retrieve the matching body patterns from the graph
def compile_query_for_rule_v2(rule_string, optional_clause=True, verbrose=True):
    #print(f'\n{rules[i]}')
    #assume these will be tab separated
    select_exp, bind_exp, filter_exp, rule_exp = rule_string.split('\t') #rules[i].split('\t')[0], rules[i].split('\t')[1], rules[i].split('\t')[2], rules[i].split('\t')[3]

    #print("select :", select_exp, len(select_exp))
    #print("bind :", bind_exp, len(bind_exp))
    #print("filter :", filter_exp, len(filter_exp))
    
    #split the rule by the following characters (<= will be the head, body separator)
    #ignores the optional blocks { } and filter not exists blocks (~{  })
    rule = re.sub(r'\t|\n|<=|\(|\)|\{|\}|,|\~', ' ', rule_exp).split()
       
    head = np.array(rule[:3])
    #print(f'>>head: {head}')
    bodys = np.array(rule[3:]).reshape(int(len(rule[3:]) / 3), 3)
    #print(f'>>bodys: {bodys}')
    
    # counting the entities in the rule body only
    i = 0
    entity_counts = dict()
    #entity_counts[head[1]] = 1
    #entity_counts[head[2]] = 1
    while i < len(bodys):
        entity_counts[bodys[i][1]] = entity_counts.get(bodys[i][1], 0) + 1
        entity_counts[bodys[i][2]] = entity_counts.get(bodys[i][2], 0) + 1
        i += 1
        
    
    bind_filter_string = ""
    if (bind_exp.startswith("BIND[") and bind_exp.endswith("]")) and len(bind_exp) > 6: #BIND[]
        clauses = str(bind_exp[5:-1]).split(',')
        for k in range(len(clauses)):
            bind_filter_string += '\nBIND ' + clauses[k] + ' .'
    if (filter_exp.startswith("FILTER[") and filter_exp.endswith("]")) and len(filter_exp) > 8: #FILTER[]
        clauses = str(filter_exp[7:-1]).split(',')
        for k in range(len(clauses)):
            bind_filter_string += '\nFILTER ' + clauses[k] + ' .'
            
    query_string = 'SELECT'
    for k in entity_counts.keys():
        if is_general_entity(k):
            #Using the Stored Procedure (DB.DBA.QName) defined by Sergio at the VOS endpoint
            # to make the proper replacement of the namespace IRI to the pre-defined prefix.
            #query_string += ' sql:QName(' + k + ')' # ' ' + k
            query_string += ' ' + k
            
    #add the specific expressions to the SELECT clause
    if (select_exp.startswith("SELECT[") and select_exp.endswith("]")) and len(select_exp) > 8: #SELECT[]
        clauses = str(select_exp[7:-1]).split(',')
        for k in range(len(clauses)):
            query_string += ' ' + clauses[k]
    
    # Splitting rule again, removed the substitution of curly braces which denote optional blocks
    if optional_clause:
        rule_body = re.sub(r'\t|\n|<=|\(|\)|,', ' ', rule_exp).split()[3:]
    else:
        rule_body = re.sub(r'\t|\n|<=|\(|\)|\{|\}|,|\~', ' ', rule_exp).split()[3:] #also remove the curly braces and '~{ }'
    #print(rule_body)
    query_string += '\nWHERE { ' 
    
    # supports upto two nested OPTIONAL blocks {.....{...}...} or {.....{...}}
    # assming that the first triple can't be optional
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

    if verbrose:
        print("\nQUERY FOR RULE BODY MATCHES:")
        print(query_string)
    
    head_string = 'SELECT'
    if is_general_entity(head[1]):
        head_string += ' ' + head[1]
    if is_general_entity(head[2]):
        head_string += ' ' + head[2]
    head_string += '\nWHERE { '
    head_string += '\n' + head[1] + ' ' + head[0] + ' '  + head[2] + ' .'
    head_string += '\n}'
    
    if verbrose:
        print("\nQUERY FOR RULE HEAD MATCHES:")
        print(head_string) 
    
    #construct the whole rule match query (including both rule body and rule head)
    aditional_head_entities = ''
    if is_general_entity(head[1]) and (head[1] not in entity_counts.keys()):
        aditional_head_entities += ' ' + head[1] 
    if is_general_entity(head[2]) and (head[2] not in entity_counts.keys()):
        aditional_head_entities += ' ' + head[2] 
    if aditional_head_entities == '':
        head_body_string = query_string[0:-1] + head_string.split('\n')[-2] + '\n}'
    else: 
        head_body_string = query_string.split('\n')[0] + aditional_head_entities + '\n' + '\n'.join(query_string.split('\n')[1:-1]) + '\n' + head_string.split('\n')[-2] + '\n}'
    if verbrose:
        print("\nQUERY FOR RULE BODY AND HEAD MATCHES:")
        print(head_body_string) 
    
    if '?X' in entity_counts.keys() and '?Y' in entity_counts.keys():
        distinct_count_string = 'SELECT (COUNT(*) AS ?count) {' + '\n' + 'SELECT DISTINCT ?X ?Y' + '\n'
    else: #ASSUME either X or Y to be present
        distinct_count_string = 'SELECT (COUNT (DISTINCT ' + ('?X' if '?X' in entity_counts.keys() else '')  + ('?Y' if '?Y' in entity_counts.keys() else '') + ') AS ?count)' + '\n'
    numerator_string = distinct_count_string + ('\n').join(head_body_string.split('\n')[1:]) 
    denominator_string = distinct_count_string + ('\n').join(query_string.split('\n')[1:]) 
    denominator_string_HC = distinct_count_string + ('\n').join(head_string.split('\n')[1:])
    if '?X' in entity_counts.keys() and '?Y' in entity_counts.keys():
        numerator_string += '\n}'
        denominator_string += '\n}'
        denominator_string_HC += '\n}'
        
    if verbrose:
        print("\nQUERY FOR NUMERATOR:")
        print(numerator_string)
    
        print("\nQUERY FOR DENOMINATOR:")
        print(denominator_string)
    
        print("\nQUERY FOR DENOMINATOR OF HEAD COVERAGE:")
        print(denominator_string_HC)
    
    return head_body_string, query_string, head_string, numerator_string, denominator_string, denominator_string_HC

###################################################################
# Evaluates a rule
###################################################################
def evaluate_rules(qe, k, i, rule, optional_clause=True, rule_metrics_file_name="./output/rule_metrics.txt"):
        if qe == None:
            qe: QueryExecutor = QueryExecutor()
            
        if not optional_clause:
            rule = re.sub(r'\{|\}', '', rule) #strip the curly braces to remove the optional blocks
            
        rs, bs, hs, numerator_query, denominator_query, denominator_query_HC = compile_query_for_rule_v2(rule, optional_clause=optional_clause, verbrose=False)
        
        denominator = int(qe.run_query_df(denominator_query).iloc[0]['count'])
        if denominator > 0:
            numerator = int(qe.run_query_df(numerator_query).iloc[0]['count'])
        else:
            numerator = 0
        
        if numerator > 0: #only if there is rule support (can introduce a SC threshold instead)
            denominator_hc = int(qe.run_query_df(denominator_query_HC).iloc[0]['count'])
            with open(rule_metrics_file_name, 'a') as f:
                f.write(f'{k:<10} {i:<10} {denominator:<10} {numerator:<10} {denominator_hc:<10}'+
                    f' {np.round(numerator/denominator if denominator>0 else 0, 8):<10}' +
                    f' {np.round(numerator/denominator_hc if denominator_hc>0 else 0, 8):<10}'+
                    f' |\t{rule}\n')
        #fp.write("%s\t%s\n" % (item,i)) 
        
            sys.stdout.flush()
            return (k, i, 1)
        else:
            return (k, i, 0)