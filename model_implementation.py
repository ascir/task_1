# Imports
import glob
import math
import os
import re
import string

from stemming.porter2 import stem

root_folder = ""  #add root folder path 

def addTerm(terms, term, stop_words):
    """
    Count the frequency of terms in a string
    ...
        Parameters:
        -----------
            terms: dict
                Dictionary to store term and term frequency
            term: string
                The term in consideration
            stop_words: list
                List of common english words to avoid
        Returns:
        --------
            terms: dict
                Updated terms dictionary
    """
    if len(term) > 2 and term not in stop_words: 
        try:
            terms[term] += 1
        except KeyError:
            terms[term] = 1
    return terms

class DocWords:
    """
    A class to represent a parsed document.
    ...
    Parameters
    ----------
        data : tuple
            Collection of docId, terms and doc_len
        docID : int
            ID of the document from the itemid
        terms : dict
            Dictionary of {index term : term frequency}
        doc_len : int
            Length of the document
    """
    def __init__(self, data, next=None):
        self.data=data
        self.next=next
        self.docID = self.data[0]
        self.terms = self.data[1]
        self.doc_len = self.data[2]
        
    def getDocId(self):
        return self.docID
    
    def getTerms(self):
        return self.terms
    
    def getDocLen(self):
        return self.doc_len
    
    # Task 3.1- Mutator method to save document length
    def setDocLen(self, length):
        self.doc_len = length

def parse_doc(xml_dir, stop_words):
    """
    Returns a collection of parsed documents
    ...
    Parameters:
    ----------
        xml_dir: string
            The directory containing the xml files we want to parse
        stop_words: list
            A list of common English words we want to skip while parsing
    Returns:
    --------
        bowColls: dict
            A collection of parsed documents in dictionary.
            Format: {docId:(docid, terms, doc_len),...}"""
    # Initialize an empty dict for storing DocWords collection
    bowColls = {}
    #Iterate through the files
    for filename in os.listdir(xml_dir):
        if filename.endswith('.xml'):  
            myfile=open(xml_dir+'/'+filename)
            # Initialize an empty dict for terms
            terms = {}
            start_end = False
            file_=myfile.readlines()
            word_count = 0 
            for line in file_:
                # Breakdown the lines
                line = line.strip()
                # Get doc ID
                if(start_end == False):
                    if line.startswith("<newsitem "):
                        for part in line.split():
                            if part.startswith("itemid="):
                                docid = part.split("=")[1].split("\"")[1]
                                break  
                    if line.startswith("<text>"):
                        start_end = True  
                elif line.startswith("</text>"):
                    break
                else:
                    line = line.replace("<p>", "").replace("</p>", "")
                    line = line.translate(str.maketrans('','', string.digits)).translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
                    line = line.replace("\\s+", " ")
                    for term in line.split():
                        word_count += 1
                        # Apply porter2 stemming
                        term = stem(term.lower()) 
                        term = term.lower() 
                        terms = addTerm(terms, term, stop_words)
            myfile.close()
            bowColls[docid] = DocWords((docid,terms,word_count), None).data
    # Collection of docs
    return(bowColls)

def parse_query(query0, stop_words):
    """
    Query parsing function
    ...
        Parameters:
        -----------
            query0: str
                A string containing a query
            stop_words: list
                List of common english words to avoid
        Returns:
        --------
            terms: dict
                {term: frequency} dictionary for terms in the query
    """
    # intialize terms dict
    terms = {}
    query0 = query0.translate(str.maketrans('','', string.digits)).translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
    query0 = query0.replace("\\s+", " ")
    for term in query0.split():
        # Apply porter2 stemming
        term = stem(term.lower()) 
        term = term.lower() 
        terms = addTerm(terms, term, stop_words)
    return terms

def calc_df(coll):
    """
    Returns a {term:df,...} dict for a document collection
    ...
        Parameters:
        -----------
            coll: dict
                Collection of documents (return of parse_docs function)
        Returns:
        --------
            df_: dict
                Term frequency in the collection of documents 
    """
    df_ = {}
    for id, doc in coll.items():
        terms = doc[1]
        for term in terms.keys():
            try:
                df_[term] += 1
            except KeyError:
                df_[term] = 1
    df_ = {k: v for k, v in sorted(df_.items(), key=lambda item: item[1], reverse=True)}
    return df_   

def tfidf(doc, df, ndocs):
    """
    Calculates the tf*idf weight for every term in the document.
    ...
        Parameters:
        -----------
        doc: dict
            A dict of {term:frequency} for a single doc
        df: dict
            Document frequency dict
        ndocs: int
            Number of documents in the doc collection
        Returns:
        --------
        weights: dict
            {term:tfif_weight} dictionary calculated based on Eq(1)
        """
    # Compute tfidf weight for each term in doc
    weights = {}
    # Compute denominator of Eq (1) for all terms in doc
    denominator = math.sqrt(sum([(math.log10(freq+1) * math.log10(ndocs/df[term]))**2 for term, freq in doc.items() if term in df]))
    for term, freq in doc.items():
        if term in df:
            # Compute numerator of Eq (1) for term in doc
            numerator = (math.log10(freq+1) * math.log10(ndocs/df[term])) 
            # Compute tfidf weight for term in doc
            weights[term] = numerator / denominator
    return weights

def tfidf_query(query_term, term_freq, N):
    """
    Calculates tf*idf weights of given query using a g(Q) equation
    ...
        Parameters:
        -----------
        query_term: string
            A string containing the query
        term_freq: dict
            Dict containing query term and frequency
        N: int
            Number of documents in collection
        Returns:
        --------
        query_weights: dict
            Tf*idf weight of each term in query
    """
    query_words = query_term.lower().split()
    query_weights = {}
    for word in query_words:
        if word in term_freq:
            query_weights[word] = math.log(1 + N/term_freq[word]) # Implementation of formula for query weight
    return query_weights

def query_weight(id, stop_words):
    """
    Parse document for the query and return tfidf of query
    ...
    Parameters:
    -----------
        id: int
            Document ID
        stop_words: list
            List of common english words to avoid
    Returns:
    --------
        line: string
            The title of the document
        weight: dict
            Query terms and their weight
    """
    dirname = 'RCV1v3'
    filename = str(id)+"newsML.xml"
    myfile=open(dirname+'/'+filename)
    file_=myfile.readlines()
    for line in file_:
        # Breakdown the lines
        line = line.strip() 
        if line.startswith("<title>"):
            line = line.replace("<title>", "").replace("</title>", "")
            l = parse_query(line, stop_words)
            weight = tfidf_query(line, l, 1)
            break
    return line, weight

def calculate_ranking(query_tf_idf_scores, doc_tf_idf_scores):
    """
    Ranks documents based on abstract model of ranking Eq(2)
    ...
    Paramters:
    ----------
    query_tf_idf_scores: dict
        TF-IDF scores of query terms
    doc_tf_idf_scores: dict
        TF-IDF scores of doc terms
    Returns:
    --------
        ranking: int
            Calculated score according to abstract model of ranking Eq(2)
    """
    ranking = 0
    for term in query_tf_idf_scores:
        if term in doc_tf_idf_scores:
            # Implementation of Eq(2) g_i(Q).f_i(D)
            ranking += query_tf_idf_scores[term] * doc_tf_idf_scores[term]   
    return ranking

def avg_doc_len(coll):
    """
    Calculates the average doc length of a collection of documents.
    Also stores document lenghth in the mutator module of the DocWords object.
    ...
    Paramters:
    ----------
    coll: dict
        Collection of documents returned by the parse_docs function
    Returns:
    --------
    avgDocLength: int
        Average length of collection of documents
    """
    totalDocLength = 0
    for id, doc in coll.items():
        doc = coll[id]
        docObj = DocWords((id, doc[1], doc[2]), None)
        docObj.setDocLen(length=doc[2]) #Calling the mutator method to save DocLength
        totalDocLength += doc[2] #Index of doc_len in DocWords object
    avgDocLength = totalDocLength/len(coll)
    return avgDocLength

def bm25(coll,query_terms,df):
    """
    Calculate documents BM25 score for a query
    Parameters:
    -----------
        coll: dict
            Collection of documents returned by the parse_docs function
        q: string
            Original query string
        df: dict
            {term:df,...} dict which is a return of the calc_df function
    Returns:
    --------
        bm25_score: dict
            Dictionary of {docID: bm25_score, ... } for all documents in collection coll
    """
    bm25_score = {}
    # Defining variables for the equation
    N = len(coll)
    ni = 0
    fi = 0
    k1 = 1.2
    b = 0.75
    k2 = 1000
    avdl = avg_doc_len(coll)
    for id, doc in coll.items():
        final_bm25 = 0
        dl = doc[2]
        doc_terms = doc[1]
        K = k1*((1-b)+b*(dl/avdl))
        for query_term in query_terms:
            qfi = query_terms[query_term]
            if query_term in df:
                ni = df[query_term]
            else:
                ni = 0
            if query_term in doc_terms:
                fi = doc_terms[query_term]
            else:
                fi = 0

            bm25_doc = (1/((ni+0.5)/(N-ni+0.5)))
            if bm25_doc != 0:
                log_bm25_doc = math.log(bm25_doc)*(((k1+1)*fi)/K+fi)*(((k2+1)*qfi)/(k2+qfi))
            else:
                log_bm25_doc = 0
            final_bm25 += log_bm25_doc
        bm25_score[id] = float(final_bm25)
    return bm25_score

def print_top_terms(term_weights):
    count_top_terms = 0
    for key, value in term_weights.items():
        print(key, ":", value)
        count_top_terms += 1
        if count_top_terms == 12:
            break

def get_narrative(query_file):
    myfile = open(query_file, 'r') # file to read
    start_end = False
    file_ = myfile.readlines()
    narr = []
    #word_count = 0 
    for line in file_:
        line = line.strip()
        if start_end == False:
            if line.startswith("<narr> Narrative:"):
                line = line.strip()
                start_end = True
                #print(line.replace("\n", " "), end="")
        elif line.startswith("</Query>"):
            start_end = False
        else:
            line = line.replace("<narr> Narrative:", "").replace("</Query>", "")
            line = line.replace("\n", " ")
            line = line.translate(str.maketrans('','', string.digits)).translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
            narr.append(line)
    narr = list(filter(lambda x: x != "", narr))
    myfile.close()
    return narr

def vector_space_model(query_terms, coll):
    """
    Calculate the cosine similarity between a query and a document using the vector space model.
 
    Parameters:
    -----------
    query: dict
        Dictionary containing term frequencies of query terms
    coll: dict
        Collection of documents returned by the parse_docs function
 
    Returns:
    --------
    cosine_similarity: float
        Cosine similarity score between the query and the document
    """
    # Calculate the dot product of the query and document vectors
    dot_product = 0
    cosine_similarity_score = {}
    for id, doc in coll.items():
        doc_terms = doc[1]
        for term in query_terms:
            if term in doc_terms:
                dot_product += query_terms[term] * doc_terms[term]
    
        # Calculate the Euclidean norm of the query vector
        query_norm = math.sqrt(sum(int(query_terms[term]) ** 2 for term in query_terms))
    
        # Calculate the Euclidean norm of the document vector
        document_norm = math.sqrt(sum(int(doc_terms[term]) ** 2 for term in doc_terms))
    
        # Calculate the cosine similarity score
        cosine_similarity = dot_product / (query_norm * document_norm)
        cosine_similarity_score[id] = float(cosine_similarity)
 
    return cosine_similarity_score

def index_docs(inputpath,stop_words):
    Index = {}    # initialize the index
    os.chdir(inputpath)
    for file_ in glob.glob("*.xml"):
        start_end = False
        for line in open(file_):
            line = line.strip()
            if(start_end == False):
                if line.startswith("<newsitem "):
                    for part in line.split():
                        if part.startswith("itemid="):
                            docid = part.split("=")[1].split("\"")[1]
                            break  
                if line.startswith("<text>"):
                    start_end = True  
            elif line.startswith("</text>"):
                break
            else:
                line = line.replace("<p>", "").replace("</p>", "")
                line = line.translate(str.maketrans('','', string.digits)).translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
                for term in line.split():
                    term = stem(term.lower())
                    if len(term) > 2 and term not in stop_words:
                        try: #similar to using if loop and else 
                            try:
                                Index[term][docid] += 1 #frequency of term is greater than 2 and not in stop words
                                                        #if term exist in doc +1 
                            except KeyError:
                                Index[term][docid]=1 #frequency 
                        except KeyError:  
                            Index[term] = {docid:1} # if term doesn't exist, just initalise 
    return Index

def likelihood_IR(I, Q): 
    """
    Calculate the liklihood IR score between a query and a document using probabilistic model.
 
    Parameters:
    -----------
    query: dict
        Dictionary containing term frequencies of query terms
    coll: dict
        Collection of documents returned by the parse_docs function
 
    Returns:
    --------
    likelihood_score: float
        liklihood IR score score between the query and the document
    """
    L={}    # L is the selected inverted list
    R={}    # R is a directionary of docId:score
    D_len={} # D_len is a directionary of docId:length
    for list in I.items():
        for id in list[1].items(): 
            R[id[0]]=1       # get all document IDs and initialize as 1
            D_len[id[0]]=0.5 # initialize a small non-zero value as it will be used as denominator
        if (list[0] in Q):     # select inverted lists based on the query
                L[list[0]]= I[list[0]]
    for q_term in Q.items(): # L may not include all query terms
        if not(q_term[0] in L):
            L[q_term[0]]={}
    
    for list in I.items():
        for id in list[1].items(): # Count term occurrences in documents
            D_len[id[0]]= D_len[id[0]] + id[1]  
    for (d, sd) in R.items():
        for (term, f) in L.items():
            if not(d in f):
                f[d]=0
            sd = sd*(f[d]/D_len[d])
        R[d] = sd
    return R

def implement_models(dataset_path, query_file, stop_words):
    datasets = []
    queries = []
    query_numbers = []
    query_narr = get_narrative(query_file)

    entries = os.listdir(dataset_path)
    for dataset in entries:
        entry_path = os.path.join(dataset_path, dataset)
        if os.path.isdir(entry_path):
            datasets.append(dataset)
    queryFile = open(query_file)
    file_ = queryFile.readlines()
    for line in file_:
        # Breakdown the lines
        line = line.strip() 
        if line.startswith("<title>"):
            line = line.replace("<title>", "").replace("</title>", "").lstrip()
            queries.append(line)
        if line.startswith("<num>"):
            line = line.replace("<num> Number: ", "").lstrip()
            query_numbers.append(line)
    
    cwd = os.getcwd()
    files = os.listdir(cwd)  # Get all the files in that directory
    print("Files in %r: %s" % (cwd, files))
    
    for i in range(0, len(datasets)):
        coll = parse_doc(f'{dataset_path}/{datasets[i]}', stop_words)
        df = calc_df(coll)
        query_terms = parse_query(queries[i], stop_words)
        indexed_docs = index_docs(f'{dataset_path}/{datasets[i]}', stop_words)
        '''
        BM25 SCORE
        '''
        bm25_scores = bm25(coll,query_terms,df)
        bm25_scores = {k: v for k, v in sorted(bm25_scores.items(), key=lambda item: item[1], reverse=True)}
        #Print Top-12 terms
        print("--------------------------------------------------------------------------------------------")
        print(f"Top 12 documents based on BM25 score for topic {query_numbers[i]}")
        print("--------------------------------------------------------------------------------------------")
        print_top_terms(bm25_scores)
        with open(f'{root_folder}Result/Baseline/Baseline_{query_numbers[i]}Ranking.dat', 'w') as f:
            for id, doc in bm25_scores.items():
                print(f"{id} {bm25_scores[id]}", file = f)
        f.close()
        '''
        VECTOR SPACE MODEL SCORE
        '''
        query_narr_terms = parse_query(query_narr[i], stop_words)
        cos_sim_score = vector_space_model(query_narr_terms, coll)
        cos_sim_score = {k: v for k, v in sorted(cos_sim_score.items(), key=lambda item: item[1], reverse=True)}
        #Print Top-12 terms
        print("--------------------------------------------------------------------------------------------")
        print(f"Top 12 documents based on VSMBM25 score for topic {query_numbers[i]}")
        print("--------------------------------------------------------------------------------------------")
        print_top_terms(cos_sim_score)
        with open(f'{root_folder}Result/vsmbm25/VSMBM25_{query_numbers[i]}Ranking.dat', 'w') as f:
            for id, doc in cos_sim_score.items():
                print(f"{id} {cos_sim_score[id]}", file = f)
        f.close()
        '''
        LIKLIHOOD MODEL SCORE
        '''
        likelihood_score = likelihood_IR(indexed_docs, query_terms)
        likelihood_score = {k: v for k, v in sorted(likelihood_score.items(), key=lambda item: item[1], reverse=True)}
        print("--------------------------------------------------------------------------------------------")
        print(f"Top 12 documents based on liklihood_ir score for topic {query_numbers[i]}")
        print("--------------------------------------------------------------------------------------------")
        print_top_terms(likelihood_score)
        with open(f'{root_folder}Result/Likelihood_IR/likelihood_{query_numbers[i]}Ranking.dat', 'w') as f:
            for id, doc in likelihood_score.items():
                print(f"{id} {likelihood_score[id]}", file = f)
        f.close()
    print("Output write complete")
          
