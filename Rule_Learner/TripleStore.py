import rdflib
import os
import pandas as pd

# load the ontology in memory: builds the RDF graph.
class TripleStore:
    #__slots__ = ('onto', 'data' 'prefix_string')
    __slots__ = ('g', 'prefix_string')
    

    def __init__(self, onto_file_path, path_in, ttl_file_names=[], rdf_format='ttl', path_out="./"):
        #onto_file_path = "./FutureSOILS-ontology_05_2023.ttl" #"./FutureSOILS-ontology.ttl"
        #path_in = "./KG_ttl_03_2023/" #"./KG_ttl_08_2023/"
        #path_out = "./"
        
        #self.onto = rdflib.Graph()
        #self.onto.parse(onto_file_path, format='ttl')
        self.g = rdflib.Graph()
        self.g.parse(onto_file_path, format='ttl')
        print("The ontology definition was loaded...")
        
        if ttl_file_names==[]:
            file_names = os.listdir(path_in)
            file_names_valid = []
            for fname in file_names:
                if fname.endswith("."+rdf_format):
                    file_names_valid.append(fname)
        else:
            file_names_valid = ttl_file_names

        print("Valid data files : ", len(file_names_valid))
        #self.data = rdflib.Graph()
        for fname in file_names_valid:       
            try:
                #self.data.parse(path_in+fname)#, format=rdf_format)
                self.g.parse(path_in+fname)#, format=rdf_format)
                print(f'{fname} was loaded successfully')
            except Exception as e:
                print(f'Issue with {fname}, {e}')
                        
        #print("All the ontology data files were loaded...")
        
        self.prefix_string = '\n'.join('PREFIX '+ns[0]+': <'+str(ns[1])+'>' for ns in self.g.namespaces())
        
    # calling builtin function Graph.triples(triple) with expanding prefixes    
    # returns generator object Graph.triples (Returns triples that match the given triple pattern)
    def get_graph_triples(self, s=None, p=None, o=None):
        if s!= None:
            s = self.g.namespace_manager.expand_curie(s)
        if p!= None:
            p = self.g.namespace_manager.expand_curie(p)
        if o!= None:
            o = self.g.namespace_manager.expand_curie(o)
        return self.g.triples((s, p, o))
           

    # simple SPARQL query to retrieve a set of triples that match with a given <p> and <o>.
    # returns rdflib.plugins.sparql.processor.SPARQLResult 
    def query_getTriples_by_predicate_object(self, p: str, o: str):
        results = self.g.query(
        #f'''{self.prefix_string}
        f'''
            SELECT ?s ?p ?o
            WHERE {{
                BIND({p} AS ?p) .
                BIND({o} AS ?o) .
                ?s ?p ?o .
        }}''')
            
        return results
    
    # simple SPARQL query to retrieve a set of triples that match with a given <s> and <p>.
    # returns rdflib.plugins.sparql.processor.SPARQLResult 
    def query_getTriples_by_subject_predicate(self, s: str, p: str):
        results = self.g.query(
        #f'''{self.prefix_string}
        f'''
            SELECT ?s ?p ?o
            WHERE {{
                BIND({s} AS ?s) .
                BIND({p} AS ?p) .
                ?s ?p ?o .
        }}''')
            
        return results
    

    # simple SPARQL query to retrieve a set of triples that match with a given <s>.
    # returns rdflib.plugins.sparql.processor.SPARQLResult 
    def query_getTriples_by_subject(self, s: str):
        results = self.g.query(
        #f'''{self.prefix_string}
        f'''
            SELECT ?s ?p ?o
            WHERE {{
                BIND({s} AS ?s) .
                ?s ?p ?o .
        }}''')
            
        return results
    
    # simple SPARQL query to retrieve a set of triples that match with a given <o>.
    # returns rdflib.plugins.sparql.processor.SPARQLResult 
    def query_getTriples_by_object(self, o: str):
        results = self.g.query(
        #f'''{self.prefix_string}
        f'''
            SELECT ?s ?p ?o
            WHERE {{
                BIND({o} AS ?o) .
                ?s ?p ?o .
        }}''')
            
        return results
    
    # simple SPARQL query to retrieve a set of triples that match with a given <p>.
    # returns rdflib.plugins.sparql.processor.SPARQLResult     
    def query_getTriples_by_predicate(self, p: str):
        results = self.g.query(
        #f'''{self.prefix_string}
        f'''
            SELECT ?s ?p ?o
            WHERE {{
                BIND({p} AS ?p) . 
                ?s ?p ?o .
        }}''')
        return results
    
    # returns rdflib.plugins.sparql.processor.SPARQLResult 
    def run_query(self, query_string: str):
        return self.g.query(query_string)
    
    def convert_to_prefix(self, entity):
        if entity is not None:
            return entity.n3(self.g.namespace_manager)
        else:
            return None
        

    def result_to_df(self, results):
        '''convert the SPAQRL query result into a pandas dataframe'''
        cols = [str(v) for v in results.vars] if isinstance(results, rdflib.plugins.sparql.processor.SPARQLResult) else None
        return pd.DataFrame([[self.convert_to_prefix(c) for c in row] for row in results], columns=cols)