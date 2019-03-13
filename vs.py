from collections import defaultdict
import math
import nltk
from nltk.corpus import stopwords
import sys
import os
import pickle
from functools import reduce


file_list = os.listdir("/home/atishay/Github/Vector-Space_Model_IR/documents")
for i in range(len(file_list)):
    file_list[i] = 'documents/'+file_list[i]
document_filenames = dict(enumerate(file_list))
# {0 : "documents/lotr.txt",
#                       1 : "documents/silmarillion.txt",
#                       2 : "documents/rainbows_end.txt",
#                       3 : "documents/the_hobbit.txt"}
print(document_filenames)
# The size of the corpus
N = len(document_filenames)

dictionary = set()

postings = defaultdict(dict)

document_frequency = defaultdict(int)

length = defaultdict(float)

characters = " .,!#$%^&*();:\n\t\\\"?!{}[]<>"

def main():
    # if os.path.isfile("index_data"):
    #     print("Loading data...")
    #     with open("index_data","rb") as f:
    #         postings = dict(pickle.load(f))
    #         print(type(postings))
    #         print(len(postings))
    #         # print(postings[0].keys())
    #         # print(postings)
    # else:
    #     print("initialize_terms_and_postings\n")
    #     print("Processing and serializing data for future use...")
    #     postings = initialize_terms_and_postings()
    # print(len(postings))
    print("initialize_terms_and_postings\n")
    initialize_terms_and_postings()
    print("initialize_document_frequencies\n")
    initialize_document_frequencies()
    print("initialize_lengths\n")
    initialize_lengths()
    while True:
        do_search()

def initialize_terms_and_postings():
    global dictionary, postings
    for id in document_filenames:
        f = open(document_filenames[id],'r')
        document = f.read()
        f.close()
        terms = tokenize(document)
        unique_terms = set(terms)
        dictionary = dictionary.union(unique_terms)
        for term in unique_terms:
            postings[term][id] = terms.count(term)
    # with open("index_data","wb") as f:
    #         pickle.dump(postings,f, protocol=pickle.HIGHEST_PROTOCOL)
    #         return postings
    # with open('posting','r') as f:
    #     for i in postings:
    #         f.write(postings[i])                                        # the value is the
                                                   # frequency of the
                                                   # term in the
                                                   # document

def tokenize(document):
    terms = document.lower().split()
    stop_words = list(stopwords.terms('english'))
    terms = [w for w in list(terms) if not w in stop_words]
    return terms

def initialize_document_frequencies():
    global document_frequency
    for term in dictionary:
        document_frequency[term] = len(postings[term])

def initialize_lengths():
    """Computes the length for each document."""
    global length
    for id in document_filenames:
        l = 0
        for term in dictionary:
            l += imp(term,id)**2
        length[id] = math.sqrt(l)

def imp(term,id):
    """Returns the importance of term in document id.  If the term
    isn't in the document, then return 0."""
    if id in postings[term]:
        return postings[term][id]*inverse_document_frequency(term)
    else:
        return 0.0

def inverse_document_frequency(term):
    """Returns the inverse document frequency of term.  Note that if
    term isn't in the dictionary then it returns 0, by convention."""
    if term in dictionary:
        return math.log(N/document_frequency[term],2)
    else:
        return 0.0

def do_search():
    """Asks the user what they would like to search for, and returns a
    list of relevant documents, in decreasing order of cosine
    similarity."""
    query = tokenize(input("Search query >> "))
    if query == []:
        sys.exit()
    # find document ids containing all query terms.  Works by
    # intersecting the posting lists for all query terms.
    relevant_document_ids = intersection(
            [set(postings[term].keys()) for term in query])
    # print(relevant_document_ids)
    if not relevant_document_ids:
        print("No documents matched all query terms.")
    else:
        scores = sorted([(id,similarity(query,id))
                         for id in relevant_document_ids],
                        key=lambda x: x[1],
                        reverse=True)
        print("Rank: filename")
        rank=1
        for (id,score) in scores:
            if rank<=10:
                print(str(rank)+": "+document_filenames[id])
            
            rank+=1
            # print str(score)+": "+document_filenames[id]
        print("total = " +rank)

def intersection(sets):
    return reduce(set.intersection, [s for s in sets])

def similarity(query,id):
    similarity = 0.0
    for term in query:
        if term in dictionary:
            similarity += inverse_document_frequency(term)*imp(term,id)
    similarity = similarity / length[id]
    return similarity

if __name__ == "__main__":
    main()
