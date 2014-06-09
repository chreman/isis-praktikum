import os, string, glob
import xml.etree.ElementTree as ET
from itertools import chain


# Import Network Analysis Tools
import networkx as nx
import igraph as ig

# Import language processing tools
from gensim import corpora, models, similarities
from nltk.corpus import stopwords
from nltk.stem.snowball import EnglishStemmer as ES
from nltk.stem.snowball import GermanStemmer as GS



def main():

    xmls = get_filelist("xml")
    docs = [ET.ElementTree(file=xml) for xml in xmls]
    texts = [[p.text for p in doc.iter("{http://rtf2xml.sourceforge.net/}para")
                if p.text != None]
                    for doc in docs]
    texts = [text for text in texts if len(text) > 0]

    #replace with MaxQDA Export
    reftexts = list(enumerate(texts))

    perform_analysis("isis-lsa", content = texts, testdata = reftexts,
                        model="lsa", depth="document",
                        num_topics=20, num_words=20, threshold=0.05)
    perform_analysis("isis-lsa", content = texts, testdata = reftexts,
                        model="lsa", depth="paragraph",
                        num_topics=20, num_words=20, threshold=0.05)
    perform_analysis("isis-lsa", content = texts, testdata = reftexts,
                        model="lsa", depth="sentence",
                        num_topics=20, num_words=20, threshold=0.05)


    perform_analysis("isis-lda", content = texts, testdata = reftexts,
                        model="lda", depth="document",
                        num_topics=20, num_words=20)
    perform_analysis("isis-lda", content = texts, testdata = reftexts,
                        model="lda", depth="paragraph",
                        num_topics=20, num_words=20)
    perform_analysis("isis-lda", content = texts, testdata = reftexts,
                        model="lda", depth="sentence",
                        num_topics=20, num_words=20)



def perform_analysis(keyword, content=None, testdata = None,
                        model="lsa", depth="document",
                        num_topics=50, num_words=50, threshold=0):
    
    try:
        dictionary, corpus = load_dictionary(keyword, depth)
    except Exception, e:
        dictionary, corpus = preprocess_content(content, keyword, depth)

    if model is "lsa":
        _model = create_lsi_model(dictionary, corpus, num_topics)
    if model is "lda":
        _model = create_lda_model(dictionary, corpus, num_topics)
    if model is "lsa":
        _model = create_lsi_model(dictionary, corpus, num_topics)
    if model is "lda":
        _model = create_lda_model(dictionary, corpus, num_topics)
    if model is "hdp":
        _model = create_hdp_model(dictionary, corpus)
    
    export_matrix(keyword, dictionary, _model, num_topics, num_words, depth)
    export_topic_list(keyword, dictionary, _model, num_topics, num_words, depth)
    export_word_graph(keyword, dictionary, _model, num_topics, num_words, threshold, depth)

    evaluate_model(keyword, testdata, _model, num_topics, num_words, threshold, depth)


def get_filelist(extension):
    """ Creates a list of files in a folder with a given extension.
    Navigate to this folder first.
    """
    return [f for f in glob.glob("articles/*.{0}".format(extension))]


def preprocess_content(content, keyword, depth="document"):
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
    stopg = stopwords.words("german")

    #stoplist can be extended like this:
    # stope.extend(["worda","wordb",...])
    stope.extend(["sustainability", "sustainable", "sustain", "sustaining",
                    "sustains", "sustained", "sustainabilities", "",
                    "et", "al"])
    stopg.extend([""])

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
                    if word not in tokens_once]
                    for text in texts]

    #create dictionary and save for later use
    dictionary = corpora.Dictionary(texts)
    dictionary.save('{0}_{1}.dict'.format(keyword, depth))

    #create corpus and save for later use
    corpus = [dictionary.doc2bow(text) for text in texts]
    corpora.MmCorpus.serialize('{0}_{1}_corpus.mm'.format(keyword, depth), corpus)

    return dictionary, corpus


def preprocess_query(query):
    """ Performs a range of preprocessing over a query string.
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
    stopg = stopwords.words("german")

    #stoplist can be extended like this:
    stope.extend([""])
    stopg.extend([""])

    query = [ES().stem(str(word.encode("utf8")).translate(None, delete_table))
                for word in query.lower().split()
                    if str(word.encode("utf8")).translate(None, delete_table) not in stope]
    return query


def load_dictionary(keyword, depth):
    """ Load dictionary and corpus from disk.
    """
    dictionary = corpora.Dictionary.load('{0}_{1}.dict'.format(keyword, depth))
    corpus = corpora.MmCorpus('{0}_{1}_corpus.mm'.format(keyword, depth))
    return dictionary, corpus


def create_lsi_model(dictionary, corpus, num_topics):
    """ Perform an analysis with an LSI-Model.
    """
    return models.LsiModel(corpus, id2word=dictionary, num_topics=num_topics)


def create_lda_model(dictionary, corpus, num_topics):
    """ Perform an analysis with an LDA-Model.
    """
    return models.LdaModel(corpus, id2word=dictionary, num_topics=num_topics)


def create_hdp_model(dictionary, corpus):
    """ Perform an analysis with an HDP-Model.
    """
    return models.HdpModel(corpus, id2word=dictionary)


def evaluate_model(keyword, testdata, model, num_topics, num_words, threshold, depth):
    """ Testdata has to be a list of tuples with form
        (topicnr, 'reference text')
    """
    dictionary, corpus = load_dictionary(keyword, depth)
    for reftopic, text in testdata:
        if len(text) > 0:
            query = preprocess_query(text)
            query_bow = dictionary.doc2bow(query)
            query_lsi = model[query_bow]
            results = sorted(query_lsi, key=lambda item: -item[1])
            evaluations = [(reftopic+1, results[0]+1) for results in results]
            true_positives = sum([1 for evaluation in evaluations if evaluation[1] == reftopic+1])
            false_negatives = sum([1 for evaluation in evaluations if evaluation[1] != reftopic+1])
            sensitivity = float(true_positives) / float(false_negatives)
            print true_positives, false_negatives, sensitivity




def export_word_graph(keyword, dictionary, model, num_topics, num_words, threshold, depth):
    """ Constructs a network of relations between words and topics.
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
    nx.write_gml(W, "{0}_{1}_{2}x{3}.gml".format(keyword, depth,
                                            num_topics, num_words))

    # read from disk as GML and create as igraph.Graph
    G = ig.read("{0}_{1}_{2}x{3}.gml".format(keyword, depth,
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
    ig.plot(gc, "{0}_{1}_{2}x{3}_FR.svg".format(keyword, depth,
                                            num_topics, num_words), **visual_style)


def export_topic_list(keyword, dictionary, model, num_topics, num_words, depth):

    with open("{0}_{1}_{2}x{3}_topics.csv".format(keyword, depth,
                                            num_topics, num_words), "w") as topics:
        topics.write("Words,Weight\n")


    with open("{0}_{1}_{2}x{3}_topics.csv".format(keyword, depth,
                                            num_topics, num_words), "a") as topics:
        n = 1
        for t in model.show_topics(num_topics, num_words, formatted=False):
            t = sorted(t, key=lambda item: -item[0])
            topics.write("topic nr {0}, enter name\n".format(str(n)))
            for word in range(num_words):
                # topicnr, wordid, word, weight
                topics.write(str(t[word][1]) +"," + str(t[word][0]) + "\n")
            n += 1
            topics.write("\n")


def export_matrix(keyword, dictionary, model, num_topics, num_words, depth):
    """ Exports the results of the LSA into gephi-usable format.
    This network is a bipartite one and needs to be transformed first,
    before you can start with other graph algorithms.
    """

    with open("{0}_{1}_{2}x{3}_nodes.csv".format(keyword, depth,
                                        num_topics, num_words), "w") as nodes:
        nodes.write("Id,Label,Partition\n")

    with open("{0}_{1}_{2}x{3}_edges.csv".format(keyword, depth,
                                        num_topics, num_words), "w") as edges:
        edges.write("Source,Target,Label,Weight\n")


    with open("{0}_{1}_{2}x{3}_nodes.csv".format(keyword, depth,
                                        num_topics, num_words), "a") as nodes:
        for item in dictionary.token2id.items():
            nodes.write(str(item[1]) + "," + str(item[0]) + "," + "Word" + "\n")
        for i in range(num_topics):
            nodes.write("{0},Topicnr {1},Topic\n".format(
                                                str(len(dictionary) + i + 1),
                                                str(i)))


    with open("{0}_{1}_{2}x{3}_edges.csv".format(keyword, depth,
                                        num_topics, num_words), "w") as edges:
        n = 0
        for t in model.show_topics(num_topics, num_words, formatted=False):
            for word in range(num_words):
                # topicnr, wordid, word, weight
                edges.write(str(len(dictionary) + n + 1) +","
                    + str(dictionary.token2id[t[word][1]]) + "," 
                    + str(t[word][1]) +","
                    + str(t[word][0]) + "\n")
            n += 1



if __name__ == "__main__":
    main()