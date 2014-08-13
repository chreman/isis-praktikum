#

import string
import glob
import time
import xml.etree.ElementTree as ET
from itertools import chain

# Import reader
import xlrd
import csv
import requests

# Import data handlers
import collections

# Import Network Analysis Tools
import networkx as nx
import igraph as ig

# Import language processing tools
from gensim import corpora, models
from nltk.corpus import stopwords
from nltk.stem.snowball import EnglishStemmer as ES



def main():
    """
    Runs a standard analysis.
    Put pdf files in an 'files' subfolder in the working
    directory, and run the script.
    """

    depth = "document"

    convert_pdfs()
    xmls = get_filelist("files", "xml")
    docs = []
    for xml in xmls:
        try:
            docs.append(ET.ElementTree(file=xml))
        except Exception, e:
            print e, xml
            continue
    print "%s documents are in the corpus." %str(len(docs))
    #docs = [ET.ElementTree(file=xml) for xml in xmls]
    texts = [[p.text for p in doc.getroot().findall(".//*[@class='DoCO:TextChunk']")
                if p.text != None]
                    for doc in docs]

    perform_analysis("isis", content = texts,
                        model="lsa", depth=depth, num_topics=110,
                        show_topics = 20, num_words=20, threshold=0)

    perform_analysis("isis", content = texts,
                        model="lda", depth=depth, num_topics = 20,
                        show_topics = 20, num_words=10)


def convert_pdfs():
    """
    Converts pdfs to xml via 
    https://gist.github.com/yoavram/4351598
    and http://pdfx.cs.man.ac.uk

    It looks for unconverted pdfs.
    """
    pdfs = get_filelist("files", "pdf")
    pdfs = set([f.rstrip(".pdf").replace(" ", "") for f in pdfs])
    xmls = get_filelist("files", "xml")
    xmls = set([f.rstrip(".xml") for f in xmls])
    filelist = pdfs - xmls
    for pdf in filelist:
        pypdfx(pdf)


def perform_analysis(keyword, content=None, testdata = None,
                        model="lsa", depth="document", num_topics = 20,
                        show_topics = 20, num_words = 20, threshold=0):
    """
    Workflow for topic analysis.

    Looks for earlier dicionary and corpus, if not creates them
    from provided documents.

    Creates either LSA or LDA model and evaluates it.

    Output: nodes and edges csv for gephi, a topic csv and 
    a network visualization.
    """
    try:
        dictionary, corpus = load_dictionary(keyword, depth)
    except Exception, e:
        dictionary, corpus = preprocess_content(content, keyword, depth)

    print "\nBeginning with analysis at %s." % time.ctime()

    if model is "lsa":
        _model = create_lsi_model(dictionary, corpus, num_topics)
    if model is "lda":
        _model = create_lda_model(dictionary, corpus, num_topics)

    testdata = load_reference_texts(model)
    evaluate_model(keyword, testdata, model, _model, num_words, threshold, depth)
    #test_for_topic_convergence(keyword, testdata, model, _model, num_topics, threshold, depth)

    export_matrix(keyword, dictionary, model, _model, show_topics, num_words, depth)
    export_topic_list(keyword, dictionary, model, _model, show_topics, num_words, depth)
    export_word_graph(keyword, dictionary, model, _model, show_topics, num_words, threshold, depth)


def get_filelist(path, extension):
    """
    Creates a list of files in a folder with a given extension.
    Navigate to this folder first.
    """
    return [f for f in glob.glob("{0}/*.{1}".format(path, extension))]


def preprocess_content(content, keyword, depth="document"):
    """
    Takes a list of documents, removes non-alphabetical characters,
    removes a list of stopwords, performs stemming and creates
    a dictionary and a corpus for this set of documents for re-use.
    """
    print "\nBeginning with preprocessing at %s." % time.ctime()

    if depth is "document":
        if type(content[0]) is list:
            documents = [" ".join(text) for text in content]
        else:
            documents = content
    if depth is "paragraph":
        documents = list(chain.from_iterable(content))
    if depth is "sentence":
        documents = list(chain.from_iterable(["".join(text).split(". ") for text in content]))

    #filter out digits and special characters
    delete_table = string.maketrans(string.ascii_lowercase,
                                    ' ' * len(string.ascii_lowercase))

    # remove common words and tokenize
    stope = stopwords.words("english")

    #stoplist can be extended like this:
    # stope.extend(["worda","wordb",...])
    with open("stopwords.csv") as stopcsv:
        reader = csv.reader(stopcsv)
        for row in reader:
            stope.extend(row)

    print "\nThis is a raw input document:"
    print documents[0]

    #texts are cleaned (characters only), filtered (stopwords removed) and stemmed (reduced to word stem)
    texts = [[ES().stem(str(word.encode("utf8")).translate(None, delete_table))
                for word in document.lower().split()
                    if str(word.encode("utf8")).translate(None, delete_table) not in stope]
                        for document in documents]

    # remove words that appear only once
    all_tokens = sum(texts, [])
    tokens_once = set(word for word in set(all_tokens)
                                        if all_tokens.count(word) == 1)
    texts = [[word for word in text
                    if word not in tokens_once and len(word) > 1]
                    for text in texts]

    print "\nThis is the raw document after cleaning, filtering, stemming and removal of unique words."
    print texts[0]

    #create dictionary and save for later use
    dictionary = corpora.Dictionary(texts)
    dictionary.save('{0}_{1}.dict'.format(keyword, depth))

    #create corpus and save for later use
    corpus = [dictionary.doc2bow(text) for text in texts]
    corpora.MmCorpus.serialize('{0}_{1}_corpus.mm'.format(keyword, depth), corpus)

    return dictionary, corpus


def preprocess_query(query):
    """
    Performs preprocessing steps for a query string.
    Removing stopword, filtering for alphabet character only,
    and stemming.
    """
    try:
        if type(query[0]) is list:
            query = [" ".join(text) for text in query]
    except Exception, e:
        pass
    if type(query) is list:
        query = " ".join(query)

    #filter out digits and special characters
    delete_table = string.maketrans(string.ascii_lowercase,
                                    ' ' * len(string.ascii_lowercase))

    # remove common words and tokenize
    stope = stopwords.words("english")

    #stoplist can be extended like this:
    with open("stopwords.csv") as stopcsv:
        reader = csv.reader(stopcsv)
        for row in reader:
            stope.extend(row)

    query = [ES().stem(str(word.encode("utf8")).translate(None, delete_table))
                for word in query.lower().split()
                    if str(word.encode("utf8")).translate(None, delete_table) not in stope]
    return query


def load_dictionary(keyword, depth):
    """
    Load dictionary and corpus from disk.
    """
    dictionary = corpora.Dictionary.load('{0}_{1}.dict'.format(keyword, depth))
    corpus = corpora.MmCorpus('{0}_{1}_corpus.mm'.format(keyword, depth))
    return dictionary, corpus


def create_lsi_model(dictionary, corpus, num_topics):
    """
    Perform an analysis with an LSI-Model.
    """
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    return models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=num_topics)


def create_lda_model(dictionary, corpus, num_topics):
    """
    Perform an analysis with an LDA-Model.
    """
    return models.LdaModel(corpus, id2word=dictionary, num_topics=num_topics)


def load_reference_texts(model):
    """
    Loads reference texts from disk.
    Reference texts should be placed in a folder in the scripts
    directory and have to be direct output from MaxQDA.
    """
    with open("testdata/{0}_codes.csv".format(model)) as codes:
        reader = csv.reader(codes)
        translation = {row[0]:int(row[1]) for row in reader}
    xls = xlrd.open_workbook("testdata/testdata.xls")
    codings = xls.sheet_by_name("Codings")
    topics = [row.value for row in codings.col(2)]
    topics = [topic for topic in topics[1:]]
    topics = [translation[topic] for topic in topics]
    texts = [row.value for row in codings.col(6)]
    testdata = zip(topics, texts)[1:]
    return testdata


def evaluate_model(keyword, testdata, modelname, model, num_words, threshold, depth):
    """
    Testdata has to be a list of tuples with form
    
        [(topicnr, 'reference text')]
    
    """
    dictionary, corpus = load_dictionary(keyword, depth)
    export_evaluation_header(keyword, depth)
    
    evaluations = []
    for ref, text in testdata:
        query = preprocess_query(text)
        query_bow = dictionary.doc2bow(query)
        query_model = model[query_bow]
        results = sorted(query_model, key=lambda item: -item[1])
        if modelname is "lsa":
            evaluation = (ref, results[1][0]+1)
        if modelname is "lda":
            evaluation = (ref, results[0][0]+1)
        evaluations.append(evaluation)
        # evaluation = (referencetopic, lsa-result)

    for ref in set(d[0] for d in testdata):
        true_positives, true_negatives = 0, 0
        false_positives, false_negatives = 0, 0
        for evaluation in evaluations:
            # # # apply test magic here # # # 
            if evaluation[0] == ref:
                if evaluation[1] == ref:
                    true_positives += 1
            if evaluation[0] == ref:
                if evaluation[1] != ref:
                    false_negatives += 1
            if evaluation[0] != ref:
                if evaluation[1] == ref:
                    false_positives += 1
            if evaluation[0] != ref:
                if evaluation[1] == ref:
                    true_negatives += 1
            # # # # # # # # # # # # # # # # #
        test_pos = true_positives + false_positives
        test_neg = false_negatives + true_negatives
        cond_pos = true_positives + false_negatives
        cond_neg = false_positives + true_negatives
        total = len(evaluations)
        if cond_pos != 0:
            recall = float(true_positives) / float(cond_pos)
        if cond_neg != 0:
            specificity = float(true_negatives) / float(cond_neg)
        else:
            specificity = 0
        if test_pos != 0:
            precision = float(true_positives) / float(test_pos)
        else:
            precision = 0
        if test_neg != 0:
            neg_pred_value = float(true_negatives) / float(test_neg)
        accuracy = float(true_positives + true_negatives) / float(total)
        print ""
        print "The confusion table for %s on %s level and topic nr. %s is:" %(keyword, depth, str(ref))
        print "TP: {0}   FN: {1} \nFP: {2}   TN: {3}".format(true_positives, false_negatives, false_positives, true_negatives)
        print "Recall: %.4f" % recall
        print "Specificity: %.4f" % specificity
        print "Precision: %.4f" % precision
        print "Neg. Predict. Value: %.4f" % neg_pred_value
        print "Accuracy: %.4f" % accuracy
        print ""
        print ""
        export_evaluation_results(keyword, depth, model, [str(ref), 
                                                            str(total), 
                                                            str(cond_pos),
                                                            str(true_positives), 
                                                            str(false_negatives), 
                                                            str(false_positives), 
                                                            str(true_negatives), 
                                                            str(recall), 
                                                            str(specificity), 
                                                            str(precision), 
                                                            str(neg_pred_value), 
                                                            str(accuracy)
                                                            ])


def test_for_topic_convergence(keyword, testdata, modelname, model, show_topics, threshold, depth):
    """
    This is experimental and not used in the analysis.
    Especially for LDA it shows a visualization of topic-mapping.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    max_id = max(td[0] for td in testdata)

    dictionary, corpus = load_dictionary(keyword, depth)
    runs = 10

    if modelname is "lsa":
        convergence = np.zeros((max_id*runs+runs, model.projection.k+1))
    if modelname is "lda":
        convergence = np.zeros((max_id*runs+runs, show_topics+1))
    
    i = 0
    while i < runs:
        evaluations = []
        if modelname is "lda":
            model = create_lda_model(dictionary, corpus, show_topics)
        for ref, text in testdata:
            query = preprocess_query(text)
            query_bow = dictionary.doc2bow(query)
            query_model = model[query_bow]
            results = sorted(query_model, key=lambda item: -item[1])
            if modelname is "lsa":
                evaluation = (ref, results[1][0]+1)
            if modelname is "lda":
                evaluation = (ref, results[0][0]+1)
            evaluations.append(evaluation)
            # evaluation = (referencetopic, lsa-result)

            conv = sorted(collections.Counter(evaluations).most_common(), key=lambda item: -item[0][0])
            for con in conv:
                if con[0][0] == ref:
                    convergence[(con[0][0]*runs)+i, con[0][1]] += con[1]
        i += 1
    row_max = np.amax(convergence+1, 1)
    convergence = convergence / row_max[:,None]
    plt.pcolor(np.log10((convergence+1)*10))
    plt.show()


def export_evaluation_header(keyword, depth):
    """
    auxiliary function
    """
    with open("{0}_{1}_evaluation.csv".format(keyword, depth), "w") as evaluation:
        evaluation.write("topicId,testdataSize,sampleSize,TP,FN,FP,TN,recall,Specificity,Precision,NegPredValue,Accuracy\n")


def export_evaluation_results(keyword, depth, model, *results):
    """
    auxiliary function
    """
    with open("{0}_{1}_evaluation.csv".format(keyword, depth), "a") as evaluation:
        evaluation.write(",".join(*results)+"\n")


def export_word_graph(keyword, dictionary, modelname, model, num_topics, num_words, threshold, depth):
    """
    Constructs a network of relations between words and topics.
    This can be seen as a bipartite network, which is then transformed
    into a unipartite network of word-word relations.
    Of this network the giant component is taken and visualized.
    """

    H = nx.Graph()
    for word in dictionary.token2id.items():
        H.add_node(word[1], text=word[0], partition=1)

    n=0
    for topic in model.show_topics(num_topics, num_words, formatted=False):
        H.add_node(len(dictionary)+n+1, partition=0)
        for word in range(num_words):
            if topic[word][0] > threshold: #only positive weights
                H.add_edge(len(dictionary)+n+1, dictionary.token2id[topic[word][1]])
        n += 1

    # construct bipartite graph with topics as 0 and words as 1
    word_nodes, topic_nodes = nx.algorithms.bipartite.sets(H)

    # create unipartite projection for words
    W = nx.algorithms.bipartite.weighted_projected_graph(H, word_nodes)

    # write to disk as GML
    nx.write_gml(W, "{0}_{1}_{2}x{3}.gml".format(keyword+modelname, depth,
                                            num_topics, num_words))

    # read from disk as GML and create as igraph.Graph
    G = ig.read("{0}_{1}_{2}x{3}.gml".format(keyword+modelname, depth,
                                            num_topics, num_words), "gml")

    # filter to giant component
    gc = ig.VertexClustering(G).giant()
    visual_style = {}
    visual_style["layout"] = G.layout_fruchterman_reingold()
    visual_style["vertex_size"] = 8
    visual_style["vertex_label"] = G.vs["text"]
    visual_style["edge_width"] = 0.5
    visual_style["bbox"] = (1200, 1200)
    visual_style["margin"] = 50
    ig.plot(gc, "{0}_{1}_{2}x{3}_FR.svg".format(keyword+modelname, depth,
                                            num_topics, num_words), **visual_style)


def export_topic_list(keyword, dictionary, modelname, model, num_topics, num_words, depth):

    with open("{0}_{1}_{2}x{3}_topics.csv".format(keyword+modelname, depth,
                                            num_topics, num_words), "w") as topics:
        topics.write("Words,Weight\n")


    with open("{0}_{1}_{2}x{3}_topics.csv".format(keyword+modelname, depth,
                                            num_topics, num_words), "a") as topics:
        n = 1
        for t in model.show_topics(num_topics, num_words, formatted=False):
            # item[0] are the correlations of the words in a topic
            t = sorted(t, key=lambda item: -item[0])
            topics.write("topic nr {0}, title\n".format(str(n)))
            for word in range(num_words):
                if t[word][0] > 0:
                    # word, weight
                    topics.write(str(t[word][1]) +"," + str(t[word][0]) + "\n")
            n += 1
            topics.write("\n")


def export_matrix(keyword, dictionary, modelname, model, show_topics, num_words, depth):
    """
    Exports the results of the LSA into gephi-usable format.
    The exported network is a bipartite one and needs to be transformed first,
    before you can start with other graph algorithms.

    Output: nodes.csv, edges.csv
    """
    # write headers
    with open("{0}_{1}_{2}x{3}_nodes.csv".format(keyword+modelname, depth,
                                        show_topics, num_words), "w") as nodes:
        nodes.write("Id,Label,Partition\n")

    with open("{0}_{1}_{2}x{3}_edges.csv".format(keyword+modelname, depth,
                                        show_topics, num_words), "w") as edges:
        edges.write("Source,Target,Label,Weight\n")

    
    with open("{0}_{1}_{2}x{3}_nodes.csv".format(keyword+modelname, depth,
                                        show_topics, num_words), "a") as nodes:
        for item in dictionary.token2id.items():
            nodes.write(str(item[1]) + "," + str(item[0]) + "," + "Word" + "\n")
        for i in range(show_topics):
            nodes.write("{0},Topicnr {1},Topic\n".format(
                                                str(len(dictionary) + i + 1),
                                                str(i)))


    with open("{0}_{1}_{2}x{3}_edges.csv".format(keyword+modelname, depth,
                                        show_topics, num_words), "w") as edges:
        n = 0
        for t in model.show_topics(show_topics, num_words, formatted=False):
            for word in range(num_words):
                # topicnr, wordid, word, weight
                edges.write(str(len(dictionary) + n + 1) +","
                    + str(dictionary.token2id[t[word][1]]) + "," 
                    + str(t[word][1]) +","
                    + str(t[word][0]) + "\n")
            n += 1


 
def pypdfx(filename):
    """
    Filename is a name of a pdf file WITHOUT the extension
    The function will print messages, including the status code,
    and will write the XML file to <filename>.xml

    source: https://gist.github.com/yoavram/4351598
    """
    url = "http://pdfx.cs.man.ac.uk"

    fin = open(filename + '.pdf', 'rb')
    files = {'file': fin}
    try:
        print 'Sending', filename, 'to', url
        r = requests.post(url, files=files, headers={'Content-Type':'application/pdf'})
        print 'Got status code', r.status_code
    finally:
        fin.close()
    fout = open(filename.replace(" ","") + '.xml', 'w')
    fout.write(r.content)
    fout.close()
    print 'Written to', filename.replace(" ","") + '.xml'



if __name__ == "__main__":
    main()