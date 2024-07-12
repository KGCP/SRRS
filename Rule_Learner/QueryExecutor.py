import rdflib
from SPARQLWrapper import SPARQLWrapper, Wrapper
import pandas as pd

# IMPORTANT: Requires secure VPN connection to ANU (Global Protect VPN)
#   And Access to the VOS endpoint if running locally
#   Using SSH tunnelling and port forwarding
# For example, the below command will allow you to access the service running on port 8890 on dirt.gpu by pointing at port 8890 on localhost .
#   user@local-desktop:~$ ssh -L 8890:localhost:8890 u1234567@dirt.gpu -J u1234567@bulwark.cecs.anu.edu.au

# Or configure the .ssh/config file and use command `ssh dirt_vos`
#   the contents of ~/.ssh/id_rsa.pub should have been already added to testgpu-01:/home/admin/.ssh/authorized_keys , 
#   as well as bulwark:/home/users/u1234567/.ssh/authorized_keys .

'''
 Host dirt_vos
 HostName dirt.gpu
 ProxyJump u1234567@bulwark.cecs.anu.edu.au
 User u4975316
 ForwardX11 yes
 IdentityFile ~/.ssh/id_rsa
 LocalForward 8890 localhost:8890
''' 


class QueryExecutor:
    __slots__ = ('g', 'sparql')
    
    def __init__(self, prefix_file_path='./FutureSOILS-base-kg.ttl', sparql_endpoint='http://localhost:8890/sparql'):
        
        # Load the prefixes strored in the FutureSOILS-base-kg.ttl file
        # The default Namespace Manager of this (empty) graph will be used to process
        # the results of the SPARQLWrapper Queries
        self.g = rdflib.Graph().parse(prefix_file_path, format='ttl')
        
        # Python wrapper around the SPARQL service to remotely execute your queries
        self.sparql = SPARQLWrapper(sparql_endpoint)
        self.sparql.setReturnFormat('json') # using JSON return format
        
        
    def run_query(self, query_string):
        '''execute the SPARQL query specified by the query_string and return the converted (json/dict) result'''
        self.sparql.setQuery(query_string)
        return self.sparql.queryAndConvert()
    
    def run_query_df(self, query_string):
        '''execute the SPARQL query specified by the query_string and return the result as a pandas dataframe'''
        self.sparql.setQuery(query_string)
        return self.result_to_df_numeric(self.sparql.queryAndConvert())
    
    
    def get_qname(self, uri_string):
        '''get the uri_string and convert to prefixed format using the Namespace Manager of the self.g graph'''
        if uri_string is not None:
            return rdflib.term.URIRef(uri_string).n3(self.g.namespace_manager)
        else:
            return None
    
    def result_to_df(self, results):
        '''convert the SPAQRL query result into a pandas dataframe'''
        cols = results['head']['vars']
        return pd.DataFrame([[None if row.get(c) is None or row.get(c).get('value')==''
                              else self.get_qname(row.get(c).get('value')) 
                                  for c in cols] for row in results["results"]["bindings"]], columns=cols)
    
    def result_to_df_numeric(self, results):
        '''convert the SPAQRL query result into a pandas dataframe (numeric values will be converted to float)'''
        column_types_dict = {"rdf:PlainLiteral": str, 
                            "xsd:string": str, 
                            "rdf:langString": str, 
                            "rdfs:Literal": str, 
                            "xsd:anyURI": str, 
                            "<geo:wktLiteral>": str,
                            "xsd:dateTime": "datetime", #"%Y-%m-%dT%H:%M:%S.%f"
                            "xsd:date": "datetime",
                            "xsd:integer": "Int64", #int, 
                            "xsd:positiveInteger": "Int64" ,#int, 
                            "xsd:nonNegativeInteger": "Int64", #int, 
                            "lm-dt:numericUnion": float,  # removing this as qudt:DECIMAL was introduced instead
                            "xsd:float": float,
                            "qudt:DECIMAL": float,
                            "xsd:decimal": float, # encountered when averaging floats in query
                            "xsd:boolean": bool}
            
        cols = results['head']['vars']
        return pd.DataFrame([[None if row.get(c) is None or row.get(c).get('value')==''                          
                          else float(row[c]['value']) if (row[c]['type']=='typed-literal' and column_types_dict.get(self.get_qname(row[c]['datatype']), str) in [float, "Int64"])  
                          else self.get_qname(row[c]['value']) if row[c]['type']=='uri'
                          else row[c]['value'] #this will catch type='literal' and other datatypes for the moment (no processing)
                              for c in cols] for row in results["results"]["bindings"]], columns=cols)

    
    def get_attributes_row(self, row, numeric_only=True):
        '''
        Returns dataframe containing attr_vals for each entity containing attributes.
        The column names will represent the attr_name of the entities.
        '''
        attr_row_df = pd.DataFrame()
        for ind, entity in zip(row.index, row):
            if entity==None:
                continue
            
            if numeric_only:
                attr = self.run_query_df(f'''
                    SELECT ?p ?o 
                    WHERE {{ 
                    {entity} ?p ?o
                    FILTER (isNumeric(?o) || datatype(?o) = qudt:DECIMAL) . 
                    }}''') 
            else:
                attr = self.run_query_df(f'''
                    SELECT ?p ?o 
                    WHERE {{ {entity} ?p ?o 
                    FILTER (isLiteral(?o)) .
                    }}''') 

            attr['p'] = ind + ":" + attr['p']     
            attr = attr.set_index('p')
            attr_row_df = pd.concat([attr_row_df, attr])
            if not(attr_row_df.index.is_unique): #do we have duplicated attributes?
                attr_row_df = attr_row_df.loc[~attr_row_df.index.duplicated(keep='first')] #this will keep the first attribute when duplicated 

        return attr_row_df.T
    
        
    
    def get_attributes_df(self, results_df, numeric_only=True):
        ''' returning numeric attributes only by default. Set numeric=False to retieve all attributes '''
        return pd.concat([self.get_attributes_row(results_df.iloc[i], numeric_only) for i in range(len(results_df))], ignore_index=True )

    
if __name__ == '__main__':      
    ##############################################################    
    ## Example 1: running a query

    #First instantiate the QueryExecutor
    data: QueryExecutor = QueryExecutor()    

    dt_counts_query = '''
    SELECT DISTINCT ?dt (COUNT(?dt) AS ?count)
    WHERE { 
    ?s ?p ?o .
    FILTER isLiteral(?o) .
    BIND (datatype(?o) AS ?dt)
    }
    GROUP BY ?dt 
    '''
    #results = data.run_query(dt_counts_query)
    #data.result_to_df_numeric(results)

    print('Data Types present in the default graph:')
    print(data.run_query_df(dt_counts_query))
    
    
    ## Example 2: running a query and retrieving the numeric attributes of the entities returned by the query result
    q = f'''
    SELECT ?A ?X ?T 
    WHERE {{ 
    ?A fs-onto:hasObservation ?X .
    ?X sosa:observedProperty fs-data:ObservedProperty-pH .
    ?X sosa:usedProcedure fs-data:MoA-1-5-Water .
    ?X sosa:phenomenonTime ?T .
    }}
    '''

    
    #print('\npH (MoA-1-5-Water) observations:')
    #results_df = data.run_query_df(q)  
    #print(results_df.head())    
    
    ## getting attributes of the first 5 rows only as this method is very inefficient
    #print('\nRetrieving numeric attributes only (literals) of the above result:')
    #print(data.get_attributes_df(results_df[0:5], numeric_only=True)) 
    
    #print('\nRetrieving all attributes (literals) of the above result:')
    #print(data.get_attributes_df(results_df[0:5], numeric_only=False))