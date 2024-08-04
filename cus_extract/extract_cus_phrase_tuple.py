from enum import Enum
import os
from pathlib import Path
import pickle
import re
import sys
import json
import time
import networkx as nx
from transformers import pipeline
# pip install neworkx
import lib.ExclusionPhraseMerger as epm
import lib.NounPhraseMerger as npm
import lib.ExclusionDetector as eh
import spacy
# spacy.prefer_gpu()
# conda install spacy=2.0.18

class DataObjNERHelper:
    def __init__(self, dataobj_path):
        # create dataobj ner pipeline
        self.classifier = pipeline("ner", model=str(dataobj_path))

    def infer_dataobj(self, text):
        # find the positions of dataobj in the text
        # output: 
        #   list of tuple: (start_char_index, end_char_index, dataobj)
        results = self.classifier(text)
        print(results)
        pos_results = []
        if len(results) < 1:
            return pos_results
        token_pt, work_pt = 0, 0
        while token_pt < len(results):
            if results[token_pt]['entity'] == 'B':  # find a dataobj
                # find the positions of this dataobj
                work_pt = token_pt + 1
                while work_pt < len(results) and results[work_pt]['entity'] == 'I':
                    work_pt += 1
                work_pt -= 1
                # record the positions (idx of char)
                start_index = results[token_pt]['start']
                end_index = results[work_pt]['end']
                pos_results.append((start_index, end_index, text[start_index: end_index]))
                token_pt = work_pt + 1
            else:
                token_pt += 1
        print(pos_results)
        return pos_results


def cleanupUnicodeErrors(term):
    # Cleanup from mistakes before... this should really be fixed during the intial parsing of the document...
    t = re.sub('\\ufffc', ' ', term)
    t = re.sub('â€œ', '', t)
    t = re.sub('â€\\u009d', '', t)
    t = re.sub('â\\u0080\\u0094', '', t)
    t = re.sub('â\\u0080\\u009d', '', t)
    t = re.sub('â\\u0080\\u009c', '', t)
    t = re.sub('â\\u0080\\u0099', '', t)
    t = re.sub('â€', '', t)
    t = re.sub('äë', '', t)
    t = re.sub('ä', '', t)
    t = re.sub('\\u0093', '', t)
    t = re.sub('\\u0092', '', t)
    t = re.sub('\\u0094', '', t)
    t = re.sub('\\u00a7', '', t)  # Section symbol
    t = re.sub('\\u25cf', '', t)  # bullet point symbol
    t = re.sub('´', '\'', t)
    t = re.sub('\\u00ac', '', t)
    t = re.sub('\\u00ad', '-', t)
    t = re.sub('\\u2211', '', t)
    t = re.sub('\\ufb01', 'fi', t)
    t = re.sub('\\uff0c', ', ', t)
    t = re.sub('\\uf0b7', '', t)
    t = re.sub('\\u037e', ';', t)
    return t


class Analytics:
    def __init__(self):
        self.dataStore = {}
        self.currentDoc = None

    def recordPolicyStatementAnalytics(self, policyStatement):
        # negation_distance and exceptImpact
        sentenceText = cleanupUnicodeErrors(policyStatement['original_sentence'])
        if 'exceptImpact' in policyStatement and policyStatement['exceptImpact']:
            if sentenceText not in self.dataStore[self.currentDoc]['exceptions']:
                self.dataStore[self.currentDoc]['exceptions'][sentenceText] = 0
            self.dataStore[self.currentDoc]['exceptions'][sentenceText] += 1
        if 'negation_distance' in policyStatement and policyStatement['negation_distance'] >= 0:
            if sentenceText not in self.dataStore[self.currentDoc]['negations']:
                self.dataStore[self.currentDoc]['negations'][sentenceText] = []
            self.dataStore[self.currentDoc]['negations'][sentenceText].append((policyStatement['negation_distance'], policyStatement['action'][1].i))
        if sentenceText not in self.dataStore[self.currentDoc]['all']:
            self.dataStore[self.currentDoc]['all'][sentenceText] = 0
        self.dataStore[self.currentDoc]['all'][sentenceText] += 1

    def startDoc(self, filename):
        self.currentDoc = filename
        self.dataStore[self.currentDoc] = {
            'performance': {
                'startTime': time.time(),
                'endTime': 0,
            },
            'negations': {},
            'exceptions': {},
            'all': {}
        }

    def endDoc(self):
        if self.currentDoc is None:
            print('Error writing end time. No current document.')
            return

        self.dataStore[self.currentDoc]['performance']['endTime'] = time.time()
        self.currentDoc = None


class AnnotationType(Enum):
    NONE = 0
    DATA_OBJ = 1
    SHARE_VERB = 2
    COLLECT_VERB = 3
    SHARE_AND_COLLECT_VERB = 4
    ENTITY = 5

    @property
    def isShareOrCollect(self):
        return self in [AnnotationType.SHARE_VERB, AnnotationType.COLLECT_VERB, AnnotationType.SHARE_AND_COLLECT_VERB]

    @property
    def isCollect(self):
        return self == AnnotationType.COLLECT_VERB

    @property
    def isData(self):
        return self == AnnotationType.DATA_OBJ

    @property
    def isEntity(self):
        return self == AnnotationType.ENTITY

    @property
    def isNotNone(self):
        return self != AnnotationType.NONE

    @property
    def isNone(self):
        return self == AnnotationType.NONE


# TODO add pass
class KeyphraseTagger:
    # "use" is a special case... We may use your information in cojunction with advertisers to blah blah blah.
    def __init__(self, dataobj_ner_path):
        self.shareVerbs = ['share', 'sell', 'provide', 'trade', 'transfer',
                           'give', 'distribute', 'disclose', 'send', 'rent',
                           'exchange', 'report', 'transmit',  # YZ: following is newly added
                           'pass', 'express', 'supply', 'display', 'publish',
                           'release', 'deliver', 'download', 'lease', 'release',
                           'view', 'show', 'forward', 'hold', 'swap',
                           'tell']
        self.collectVerbs = ['collect', 'check', 'know', 'use', 'obtain',
                             'access', 'receive', 'gather', 'store', 'save',  # YZ: following is newly added
                             'know', 'keep', 'add', 'combine', 'record',
                             'retain', 'get', 'perform', 'analyze', 'recognize',
                             'process', 'track', 'log', 'remember', 'relate',
                             'create', 'ask', 'conduct', 'monitor', 'request',
                             'link', 'associate', 'solicit', 'read', 'preserve',
                             'contain', 'include']
        self.dataobj_ner = DataObjNERHelper(dataobj_ner_path)

    def getTag(self, token):
        def isShareVerb(self, token):
            return token.pos == spacy.symbols.VERB and token.lemma_ in self.shareVerbs

        def isCollectVerb(self, token):
            return token.pos == spacy.symbols.VERB and token.lemma_ in self.collectVerbs

        # TODO do we really want "service|app|application" here? And not check if it is a subject or how related to the verb?
        def isEntity(self, token):
            return True if token.text.lower() in ['we', 'i', 'us', 'me', 'you'] or token.ent_type_ in ['PERSON', 'ORG'] else False

        def isDataObject(self, token):  # TODO do we want to allow multi-token matches or just merge?
            return token.ent_type_ == 'DATA' and token.pos != spacy.symbols.VERB

        #############################
        if isShareVerb(self, token):
            return AnnotationType.SHARE_VERB
        elif isCollectVerb(self, token):
            return AnnotationType.COLLECT_VERB
        elif isDataObject(self, token):
            return AnnotationType.DATA_OBJ
        elif isEntity(self, token):
            return AnnotationType.ENTITY
        return AnnotationType.NONE

    def tagSentence(self, sentence):
        res = {} 
        # YZ: add extra dataobj ner results
        dataobj_ner_results = self.dataobj_ner.infer_dataobj(sentence.text)
        for token in sentence:
            tag = self.getTag(token)
            for dataobj_ner_result in dataobj_ner_results:  # YZ TODO: check token idx
                if dataobj_ner_result[0] <= token.idx < dataobj_ner_result[1]:
                    tag = AnnotationType.DATA_OBJ
                    break
            if tag.isNotNone:
                res[(token.i, token)] = tag
        return res


# TODO Refactor -- these should all be instance methods, so you don't need to keep passing common objects (graph, sentence, tokenTags)
class DependencyGraphConstructor:
    @staticmethod
    def getConjugatedVerbs(sentence, targetTok=None):
        def isComma(token):
            return token.pos_ == 'PUNCT' and token.text == ','

        def isCConj(token):
            return token.pos == spacy.symbols.CCONJ and token.lemma_ in ['and', 'or', 'nor']

        def isNegation(token):
            return token.dep == spacy.symbols.neg

        def getConjugatedVerbsInternal(results, token):
            if token.pos == spacy.symbols.VERB:
                results.append(token)
            for tok in token.children:
                if tok.i < token.i:  # Ensure we only look at children that appear AFTER the token in the sentence
                    continue
                if tok.dep == spacy.symbols.conj and tok.pos == spacy.symbols.VERB:
                    if not getConjugatedVerbsInternal(results, tok):
                        return False
                elif not (isComma(tok) or isCConj(tok) or isNegation(tok)):
                    return False
            return True

        def isTokenContainedIn(token, conjugatedVerbs):
            for vbuffer in conjugatedVerbs:
                if token in vbuffer:
                    return True
            return False

        conjugatedVerbs = []
        vbuffer = []
        for token in sentence:
            if token.pos == spacy.symbols.VERB:
                # Make sure we didn't already cover the verb...
                if isTokenContainedIn(token, conjugatedVerbs):
                    continue

                vbuffer = []
                getConjugatedVerbsInternal(vbuffer, token)
                if len(vbuffer) > 1:
                    conjugatedVerbs.append(vbuffer)

        if targetTok is not None:
            for vbuffer in conjugatedVerbs:
                if targetTok in vbuffer:
                    return vbuffer
            return []
        return conjugatedVerbs

    @staticmethod
    def getRootNodes(graph):
        def hasNoInEdges(graph, node):
            return len([n for n in graph.in_edges(node)]) == 0
        root = [n for n in graph.nodes if hasNoInEdges(graph, n)]
        return root  # Could be multiple trees...

    @staticmethod
    def collapseConjugatedEntities(graph, sentence, tokenTags):
        def traverseDownward(graph, node):
            outEdges = [dst for src, dst in graph.out_edges(node)]  # Treat it as a stack instead...
            while len(outEdges) > 0:
                n = outEdges.pop()
                if graph[node][n]['label'] == 'conj' and node[2] == n[2] and node[2] in [AnnotationType.DATA_OBJ, AnnotationType.ENTITY]:
                    # Remove link from X --> Y
                    graph.remove_edge(node, n)
                    # Replace node...
                    graph.nodes[node]['lemma'] = '{},{}'.format(graph.nodes[node]['lemma'], graph.nodes[n]['lemma'])
                    graph.nodes[node]['lemmaList'].extend(graph.nodes[n]['lemmaList'])
                    graph.nodes[node]['label'] = '{}({}) - {}'.format(node[2], graph.nodes[node]['lemma'], node[1].i)
                    outEdges2 = [e for e in graph.out_edges(n)]
                    # Add all out links from Y --> Z to X --> Z (return all nodes, so we can add to outEdges...)
                    for src, dst in outEdges2:
                        graph.add_edge(node, dst, label=graph[src][dst]['label'])
                        graph.remove_edge(src, dst)
                        outEdges.append(dst)
                    graph.remove_node(n)
                    continue
                traverseDownward(graph, n)
        ##################
        # Get root node...
        roots = DependencyGraphConstructor.getRootNodes(graph)
        for r in roots:
            traverseDownward(graph, r)

    @staticmethod
    def getNodeAnnotationTag(node):
        return node[2]

    @staticmethod
    def isVerb(graph, node):
        return graph.nodes[node]['pos'] == 'VERB'

    @staticmethod
    def areAnnotationTagsEqual(node1, node2):
        t1 = DependencyGraphConstructor.getNodeAnnotationTag(node1)
        t2 = DependencyGraphConstructor.getNodeAnnotationTag(node2)
        return t1 == t2 or t1.isShareOrCollect and t2.isShareOrCollect

    @staticmethod
    def collapseConjugatedVerbs(graph, sentence, tokenTags):
        def getNewTag(n1, n2):
            if n2[2] != AnnotationType.SHARE_AND_COLLECT_VERB and n1[2].isNotNone and n1[2] != n2[2]:
                if n2[2].isNone:
                    return n1[2]
                elif (n1[2] == AnnotationType.SHARE_VERB and n2[2] == AnnotationType.COLLECT_VERB) or (n1[2] == AnnotationType.COLLECT_VERB and n2[2] == AnnotationType.SHARE_VERB) or n1[2] == AnnotationType.SHARE_AND_COLLECT_VERB:
                    return AnnotationType.SHARE_AND_COLLECT_VERB
            return n2[2]

        def addNewVerbNode(graph, node1, node2, docStart, docEnd):
            newTag = getNewTag(node1, node2)  # Get new annotation tag
            newKey = (node2[0], node2[1], newTag)  # FIXME this doesn't really represent the updated tag...
            negation = graph.nodes[node2]['neg']  # CHECKME can node1 ever be negated if node2 is not?
            newLemma = '{},{}'.format(graph.nodes[node1]['lemma'], graph.nodes[node2]['lemma'])
            newNodeLabel = '{}({}{})'.format(newTag, newLemma, ' - NOT' if negation else '')
            newLemmaList = []
            newLemmaList.extend(graph.nodes[node1]['lemmaList'])
            newLemmaList.extend(graph.nodes[node2]['lemmaList'])
            if newKey != node2:
                graph.add_node(newKey, label=newNodeLabel, lemma=newLemma, lemmaList=newLemmaList, tag=newTag, dep=node2[1].dep_, pos=node2[1].pos_, neg=negation, docStart=docStart, docEnd=docEnd)
                return (newKey, True)
            graph.nodes[node2]['lemma'] = newLemma
            graph.nodes[node2]['label'] = newNodeLabel
            graph.nodes[node2]['neg'] = negation
            graph.nodes[node2]['lemmaList'] = newLemmaList
            graph.nodes[node2]['tag'] = newTag
            graph.nodes[node2]['startDoc'] = docStart
            graph.nodes[node2]['endDoc'] = docEnd
            return (node2, False)

        ######################################
        # Let's just walk the damn graph...
        def traverseDownward(graph, node):
            outEdges = [dst for src, dst in graph.out_edges(node)]  # Treat it as a stack instead...
            while len(outEdges) > 0:
                n = outEdges.pop()
                if graph[node][n]['label'] == 'conj' and DependencyGraphConstructor.areAnnotationTagsEqual(node, n) and DependencyGraphConstructor.isVerb(graph, node) and DependencyGraphConstructor.isVerb(graph, n):
                    # TODO the key changes due to the annotation tag potentially changing...
                    # TODO ensure separation
                    nodeTok = node[1]
                    nodeChildTok = n[1]
                    if nodeChildTok in DependencyGraphConstructor.getConjugatedVerbs(sentence, targetTok=nodeTok):
                        # Remove link from X --> Y
                        graph.remove_edge(node, n)
                        # Get new Tag
                        newTag = getNewTag(node, n)
                        if newTag == node:
                            graph.nodes[node]['lemma'] = '{},{}'.format(graph.nodes[node]['lemma'], graph.nodes[n]['lemma'])
                            graph.nodes[node]['lemmaList'].extend(graph.nodes[n]['lemmaList'])
                            graph.nodes[node]['label'] = '{}({}) - {}'.format(node[2], graph.nodes[node]['lemma'], node[1].i)
                            # Add all out links from Y --> Z to X --> Z (return all nodes, so we can add to outEdges...)
                            for src, dst in graph.out_edges(n):
                                graph.add_edge(node, dst, label=graph[src][dst]['label'])
                                graph.remove_edge(src, dst)
                                outEdges.append(dst)
                            graph.remove_node(n)
                        else:
                            # Add new tag...
                            startDoc = nodeTok.i if nodeTok.i < nodeChildTok.i else nodeChildTok.i
                            endDoc = nodeTok.i if nodeTok.i > nodeChildTok.i else nodeChildTok.i
                            newNode, addedNewNode = addNewVerbNode(graph, n, node, startDoc, endDoc)

                            if addedNewNode:
                                # Add in edges
                                for s, t in list(graph.in_edges(node)):
                                    graph.add_edge(s, newNode, label=graph[s][t]['label'])
                                    graph.remove_edge(s, t)

                                # Add out edges
                                for s, t in list(graph.out_edges(node)):
                                    graph.add_edge(newNode, t, label=graph[s][t]['label'])
                                    graph.remove_edge(s, t)

                            if not addedNewNode:
                                newNode = node

                            # Add all out links from Y --> Z to X --> Z (return all nodes, so we can add to outEdges...)
                            for src, dst in list(graph.out_edges(n)):
                                graph.add_edge(newNode, dst, label=graph[src][dst]['label'])
                                graph.remove_edge(src, dst)
                                outEdges.append(dst)

                            # Remove node from graph
                            if addedNewNode:
                                graph.remove_node(node)
                            node = newNode
                            graph.remove_node(n)
                        continue
                traverseDownward(graph, n)
        roots = DependencyGraphConstructor.getRootNodes(graph)
        for r in roots:
            traverseDownward(graph, r)

    @staticmethod
    def isVerbNegated(token, sentence):
        def isVerbNegatedInternal(token):
            return any(t.dep == spacy.symbols.neg for t in token.children)

        if isVerbNegatedInternal(token):
            return True

        # Check if verb is part of conjugated verb phrase, if so, check if any of those are negated
        conjugatedVerbs = DependencyGraphConstructor.getConjugatedVerbs(sentence, token)
        for tok in conjugatedVerbs:
            if isVerbNegatedInternal(tok):
                return True

        # Check if verb is xcomp, if so check if prior verb is negated?
        # TODO should also do advcl
        if token.dep == spacy.symbols.xcomp or token.dep == spacy.symbols.advcl:
            return DependencyGraphConstructor.isVerbNegated(token.head, sentence)
        return False

    @staticmethod
    def pruneUnattachedNodes(graph):
        def pruneChildren(graph, node):
            for s, t in list(graph.out_edges(node)):
                pruneChildren(graph, t)
            if node in graph.nodes:
                graph.remove_node(node)

        def removeNodes(graph, nodesToRemove):
            for node in nodesToRemove:
                pruneChildren(graph, node)

        def hasNoOutEdges(graph, node):
            return len([n for n in graph.out_edges(node)]) == 0

        def hasNoInEdges(graph, node):
            return len([n for n in graph.in_edges(node)]) == 0

        def doesGraphContainVerb(graph, root):
            if root[2].isShareOrCollect:
                return True
            for _, n in graph.out_edges(root):
                if doesGraphContainVerb(graph, n):
                    return True
            return False

        nodesToRemove = [node for node in graph.nodes if hasNoOutEdges(graph, node) and hasNoInEdges(graph, node)]
        removeNodes(graph, nodesToRemove)

        # Let's prune graphs that have no verbs...
        potentialRoots = [node for node in graph.nodes if hasNoInEdges(graph, node)]
        if len(potentialRoots) > 1:
            subGraphsToPrune = [r for r in potentialRoots if not doesGraphContainVerb(graph, r)]
            if len(subGraphsToPrune) < len(potentialRoots) and len(subGraphsToPrune) > 0:
                removeNodes(graph, subGraphsToPrune)

    @staticmethod
    def pruneNonSharingVerbs(graph):
        def getHead(graph, node):
            parents = [src for src, _ in graph.in_edges(node)]
            return parents[0] if len(parents) > 0 else node

        def subTreeContainsLabeledTags(graph, node, checkMatch=False):
            if checkMatch and node[2].isNotNone:
                return True
            for _, dst in graph.out_edges(node):
                if subTreeContainsLabeledTags(graph, dst, True):
                    return True
            return False

        def childrenContainDataPractice(node):  # One of the descendents need to contain a share verb...
            if (node[1].pos == spacy.symbols.VERB and node[2].isShareOrCollect and subTreeContainsLabeledTags(graph, node)):
                return True
#           elif(node[1].pos == spacy.symbols.VERB and node[1].dep_ == 'relcl'):# Only IF THE HEAD IS A DATA OBJECT OR ENTITY...
#               n2 = getHead(graph, node)[2]
#               if n2.isData or n2.isEntity:
#                   return True
            for s, child in graph.out_edges(node):
                if childrenContainDataPractice(child):
                    return True
            return False

        def pruneChildren(graph, node):
            for s, t in list(graph.out_edges(node)):
                pruneChildren(graph, t)
            if node in graph.nodes:
                graph.remove_node(node)

        def removeNodes(graph, nodesToRemove):
            for node in nodesToRemove:
                pruneChildren(graph, node)

        def hasNoOutEdges(graph, node):
            return len([n for n in graph.out_edges(node)]) == 0

        def hasNoInEdges(graph, node):
            return len([n for n in graph.in_edges(node)]) == 0

        #############################
        nodesToRemove = [node for node in graph.nodes if node[1].pos == spacy.symbols.VERB and not childrenContainDataPractice(node)]
        removeNodes(graph, nodesToRemove)

        nodesToRemove = [node for node in graph.nodes if hasNoOutEdges(graph, node) and hasNoInEdges(graph, node)]
        removeNodes(graph, nodesToRemove)

        nodesToRemove = [node for node in graph.nodes if hasNoOutEdges(graph, node) and node[2].isNone and node[1].dep not in [spacy.symbols.nsubj, spacy.symbols.dobj, spacy.symbols.nsubjpass]]
        while len(nodesToRemove) > 0:
            removeNodes(graph, nodesToRemove)
            nodesToRemove = [node for node in graph.nodes if hasNoOutEdges(graph, node) and node[2].isNone and node[1].dep not in [spacy.symbols.nsubj, spacy.symbols.dobj, spacy.symbols.nsubjpass]]

        ########################

    @staticmethod
    def convertDTreeToNxGraph(sentence, tokenTags):
        def addNode(key, node, graph, sentence):
            if key not in graph:
                negation = False
                if key[2].isShareOrCollect:
                    negation = DependencyGraphConstructor.isVerbNegated(node, sentence)
                    graph.add_node(key, label='{}({}{}) - {}'.format(key[2], node.lemma_, ' - NOT' if negation else '', node.i), tag=key[2], lemma=node.lemma_, lemmaList=[node.lemma_ if node.lemma_ != '-PRON-' else node.text.lower()], dep=node.dep_, pos=node.pos_, neg=negation, docStart=node.i, docEnd=node.i)
                else:
                    graph.add_node(key, label='{}({}) - {}'.format(key[2], node.lemma_, node.i), tag=key[2], lemma=node.lemma_, lemmaList=[node.lemma_ if node.lemma_ != '-PRON-' else node.text.lower()], dep=node.dep_, pos=node.pos_, neg=negation, docStart=node.i, docEnd=node.i)

        def convertDTreeToNxGraphInternal(root, graph, tokenTags, sentence):
            rkey = DependencyGraphConstructor.getKey(root, tokenTags)

            if rkey not in graph:
                addNode(rkey, root, graph, sentence)

            for c in root.children:
                ckey = DependencyGraphConstructor.getKey(c, tokenTags)
                if ckey not in graph:
                    addNode(ckey, c, graph, sentence)

                graph.add_edge(rkey, ckey, label=c.dep_)
                convertDTreeToNxGraphInternal(c, graph, tokenTags, sentence)
        ##############
        dgraph = nx.DiGraph()
        convertDTreeToNxGraphInternal(sentence.root, dgraph, tokenTags, sentence)
        return dgraph

    @staticmethod
    def drawGraph(g, filename):
        # try:
        #     A = nx.drawing.nx_agraph.to_agraph(g)
        #     A.draw(filename, prog='dot', args='-Granksep=2.0')
        # except:# FIXME unicode error here for some reason...
        #     pass
        return

    @staticmethod
    def getKey(root, tokenTags):
        tKey = (root.i, root)
        tag = AnnotationType.NONE if tKey not in tokenTags else tokenTags[tKey]
        return (root.i, root, tag)

    @staticmethod
    def getSimplifiedDependencyGraph(sentence, tokenTags):
        def getPathBetweenNodes(g, itok, jtok, tokenTags):
            pathNodes = nx.shortest_path(g.to_undirected(), DependencyGraphConstructor.getKey(itok, tokenTags), DependencyGraphConstructor.getKey(jtok, tokenTags))
            return g.subgraph(pathNodes).copy()

        ##############################
        if len(tokenTags) <= 1:  # Need two or more tokens...
            return None

        g = DependencyGraphConstructor.convertDTreeToNxGraph(sentence, tokenTags)
        graphs = []
        taggedTokens = [(token, tokenTags[(token.i, token)]) for i, token in tokenTags]
        for i, (itok, itag) in enumerate(taggedTokens):
            for j, (jtok, jtag) in enumerate(taggedTokens[i+1:]):
                graphs.append(getPathBetweenNodes(g, itok, jtok, tokenTags))

        # Do not prune subjects and objects...
        # TODO is it just share verbs or all?
        for i, (itok, itag) in enumerate(taggedTokens):
            if itag.isShareOrCollect:
                for _, dst in g.out_edges(DependencyGraphConstructor.getKey(itok, tokenTags)):
                    if dst[1].dep in [spacy.symbols.dobj, spacy.symbols.nsubj, spacy.symbols.nsubjpass] and dst[2].isNone:
                        graphs.append(getPathBetweenNodes(g, itok, dst[1], tokenTags))

        #################################

        g = nx.compose_all(graphs)
        DependencyGraphConstructor.collapseConjugatedVerbs(g, sentence, tokenTags)
        # Prune non-attached nodes...
        DependencyGraphConstructor.pruneUnattachedNodes(g)
        DependencyGraphConstructor.collapseConjugatedEntities(g, sentence, tokenTags)
        DependencyGraphConstructor.pruneNonSharingVerbs(g)
        #DependencyGraphConstructor.drawGraph(g, 'simplified_graph.png')
        return g


class PolicyTransformer:
    # TODO refactor so that these are instance methods
    # implicit everyone rule
    @staticmethod
    def applyPolicyTransformationRules(policyStatements, analyticsObj):
        def addPolicies(entity, collect, dataObjects, original_sentence, simplifiedStatements, actionLemma):
            # FIXME should not get a token at this point...
            if (type(entity) == str and entity == 'you') or (type(entity) == spacy.tokens.token.Token and entity.text == 'you'):
                return
            for d in dataObjects:

                simplifiedStatements.append((cleanupUnicodeErrors(entity), cleanupUnicodeErrors(collect), cleanupUnicodeErrors(d), cleanupUnicodeErrors(original_sentence), cleanupUnicodeErrors(actionLemma)))

        def addPoliciesByEntities(entities, collect, dataObjects, original_sentence, simplifiedStatements, actionLemma):
            if entities is not None and len(entities) > 0:
                if type(entities) == list:
                    for e in entities:
                        addPolicies(e, collect, dataObjects, original_sentence, simplifiedStatements, actionLemma)
                else:
                    addPolicies(entities, collect, dataObjects, original_sentence, simplifiedStatements, actionLemma)
            else:
                addPolicies('third_party_implicit', collect, dataObjects, original_sentence, simplifiedStatements, actionLemma)

        def getAgentText(agent):
            if agent is None:  # TODO CHECKME: Should we really have an implicit first party
                return 'we_implicit'
            if type(agent) == str:
                return agent
            if type(agent[1]) == str:
                return agent[1]
            return agent[1].lemma_ if agent[1].lemma_ != '-PRON-' else agent[1].text.lower()
            # return agent[1] if type(agent[1]) == unicode else agent[1].text.lower() #This needs to be the lemma unless -PRON-

        def handleShareVerb(pstatement, actionLemma, simplifiedStatements):
            agents = [getAgentText(a) for a in pstatement['agent']]
            original_sentence = pstatement['original_sentence']
            # Ensure that we don't create a conflict...
            # For example, if we have a sentence that claims "we do not collect or share X.", do not assume first party collect
            if pstatement['action'][2] == AnnotationType.SHARE_AND_COLLECT_VERB and pstatement['is_negated']:
                pass  # FIXME clean up condition check
            else:
                addPoliciesByEntities(agents, 'collect', pstatement['data_objects'], original_sentence, simplifiedStatements, actionLemma)

            # If it is "you share/not share" and entities is nil, do not assume third-party
            if len(agents) == 1 and type(agents[0]) == str and agents[0] == 'you':
                if pstatement['entities'] is None or len(pstatement['entities']) == 0:
                    pstatement['entities'] = ['we_implicit']

            collect = 'not_collect' if pstatement['is_negated'] else 'collect'
            # not sell, (you) not provide, trade, rent, exchange,
            # collect = u'collect' if actionLemma in [u'sell', u'rent', u'trade', u'exchange'] else collect
            # # you do not provide us does not mean not collect necessarily...
            # if len(agents) == 1 and type(agents[0]) == unicode and agents[0] == u'you' and actionLemma in [u'provide', u'give']:
            #     return

            addPoliciesByEntities(pstatement['entities'], collect, pstatement['data_objects'], original_sentence, simplifiedStatements, actionLemma)

        def handleCollectVerb(pstatement, actionLemma, simplifiedStatements):
            agents = [getAgentText(a) for a in pstatement['agent']]
            collect = 'not_collect' if pstatement['is_negated'] else 'collect'

            if pstatement['is_negated'] and actionLemma == 'use':
                return

            # not use, store, save. "Use" is typically conditional, so ignore negation (e.g., not use for...)
            collect = 'collect' if actionLemma in ['store', 'save'] else collect

            original_sentence = pstatement['original_sentence']
            addPoliciesByEntities(agents, collect, pstatement['data_objects'], original_sentence, simplifiedStatements, actionLemma)

        simplifiedStatements = []
        # Array of statements
        for pstatement in policyStatements:
            # TODO analytics... exceptImpact, negation_distance
            analyticsObj.recordPolicyStatementAnalytics(pstatement)

            # Get the lemmas and do it this way instead...
            for actionLemma in pstatement['action_lemmas']:
                # if actionLemma in ['share', 'sell', 'provide', 'trade', 'transfer', 'give', 'distribute', 'disclose', 'send', 'rent', 'exchange', 'report', 'transmit']:  # TODO refactor
                #     handleShareVerb(pstatement, actionLemma, simplifiedStatements)
                # elif actionLemma in ['collect', 'check', 'know', 'use', 'obtain', 'access', 'receive', 'gather', 'store', 'save']:  # TODO refactor
                #     handleCollectVerb(pstatement, actionLemma, simplifiedStatements)
                if actionLemma in ['share', 'sell', 'provide', 'trade', 'transfer',
                                   'give', 'distribute', 'disclose', 'send', 'rent',
                                   'exchange', 'report', 'transmit',
                                   'pass', 'express', 'supply', 'display', 'publish',  # YZ: newly-added
                                   'release', 'deliver', 'download', 'lease', 'release',
                                   'view', 'show', 'forward', 'hold', 'swap',
                                   'tell']:  # TODO refactor
                    handleShareVerb(pstatement, actionLemma, simplifiedStatements)
                elif actionLemma in ['collect', 'check', 'know', 'use', 'obtain',
                                     'access', 'receive', 'gather', 'store', 'save',  # YZ: newly-added
                                     'know', 'keep', 'add', 'combine', 'record',
                                     'retain', 'get', 'perform', 'analyze', 'recognize',
                                     'process', 'track', 'log', 'remember', 'relate',
                                     'create', 'ask', 'conduct', 'monitor', 'request',
                                     'link', 'associate', 'solicit', 'read', 'preserve',
                                     'contain', 'include']:  # TODO refactor
                    handleCollectVerb(pstatement, actionLemma, simplifiedStatements)

            # if pstatement['action'][2] in [AnnotationType.SHARE_VERB, AnnotationType.SHARE_AND_COLLECT_VERB]:
            #     handleShareVerb(pstatement, simplifiedStatements)
            # if pstatement['action'][2] in [AnnotationType.COLLECT_VERB, AnnotationType.SHARE_AND_COLLECT_VERB]:
            #     handleCollectVerb(pstatement, simplifiedStatements)
        return list(set(simplifiedStatements))

    @staticmethod
    def handleExceptions(policyStatements, tags):
        def clonePolicyStatement(pol):
            return {'data_objects': pol['data_objects'], 'entities': pol['entities'], 'agent': pol['agent'], 'action': pol['action'], 'action_lemmas': pol['action_lemmas'], 'is_negated': pol['is_negated'], 'negation_distance': pol['negation_distance'], 'original_sentence': pol['original_sentence'], 'exceptions': pol['exceptions']}

        def lemmatize(tokens):
            return ' '.join(t.lemma_ for t in tokens)

        def getRelevantTags(e, tags):
            return {(term.i, term): tags[(term.i, term)] for term in e if (term.i, term) in tags}

        def isAllData(tags):
            return all(tags[k].isData for k in tags)

        def isAllEntity(tags):
            return all(tags[k].isEntity for k in tags)

        newStatements = []
        removePolicyStatements = []
        for pol in policyStatements:
            if pol['exceptions'] is not None and len(pol['exceptions']) > 0:

                # Get all exceptions at first that can be resolved with keywords or all data and entity
                excepts = [(v, e) for v, e in pol['exceptions']]
                for v, e in pol['exceptions']:

                    # Record how often exceptions affect policy statements...
                    relTags = getRelevantTags(e, tags)
                    elemma = lemmatize(e)
                    if re.search(r'^.*\b(consent|you\sagree|your\s(express\s)?permission|you\sprovide|opt([\s\-](out|in))?|respond\sto\syou(r)?|disclose\sin\sprivacy\spolicy|follow(ing)?\scircumstance|permit\sby\schildren\'s\sonline\sprivacy\sprotection\sact)\b.*$', elemma):
                        # Only do the exceptions in negative cases...
                        # For example, we do not want to reverse: "We collect your personal information without your consent"
                        if not pol['is_negated']:
                            continue
                        newPol = clonePolicyStatement(pol)
                        newPol['is_negated'] = not newPol['is_negated']
                        newPol['exceptions'] = None  # TODO do we ever really need this again?
                        newPol['exceptImpact'] = True
                        newStatements.append(newPol)
                        excepts.remove((v, e))
                        removePolicyStatements.append(pol)
                    elif elemma in ['require by law', 'we receive subpoena', 'law']:
                        # Only do the exceptions in negative cases...
                        # For example, we do not want to reverse: "We collect your personal information without your consent"
                        if not pol['is_negated']:
                            continue
                        newPol = clonePolicyStatement(pol)
                        newPol['entities'] = ['government agency']
                        newPol['is_negated'] = not newPol['is_negated']
                        newPol['exceptions'] = None  # TODO do we ever really need this again?
                        newPol['exceptImpact'] = True
                        newStatements.append(newPol)
                        excepts.remove((v, e))
                        removePolicyStatements.append(pol)
                    elif len(relTags) == len(e):
                        newPol = clonePolicyStatement(pol)
                        newPol['is_negated'] = not newPol['is_negated']
                        newPol['exceptions'] = None  # TODO do we ever really need this again?
                        newPol['exceptImpact'] = True
                        # If ALL data items
                        if isAllData(relTags):
                            newPol['data_objects'] = [data.lemma_ for index, data in relTags]
                            newStatements.append(newPol)
                            excepts.remove((v, e))
                            # removePolicyStatements.append(pol)
                        # If ALL entities
                        elif isAllEntity(relTags):
                            if newPol['action'][2].isCollect:
                                newPol['agent'] = [data.lemma_ for index, data in relTags]
                            else:
                                newPol['entities'] = [data.lemma_ for index, data in relTags]
                            newStatements.append(newPol)
                            excepts.remove((v, e))
                            # removePolicyStatements.append(pol)
                    else:  # Not sure what it is, let's flip it anyway...
                        if not pol['is_negated']:
                            continue
                        newPol = clonePolicyStatement(pol)
                        newPol['is_negated'] = not newPol['is_negated']
                        newPol['exceptImpact'] = True
                        newPol['exceptions'] = None  # TODO do we ever really need this again?
                        newStatements.append(newPol)
                        excepts.remove((v, e))
                        removePolicyStatements.append(pol)
        for pol in newStatements:
            policyStatements.append(pol)
        for pol in removePolicyStatements:
            if pol in policyStatements:
                policyStatements.remove(pol)
        return policyStatements



class MixExtractor:
    def __init__(self, nlpModel, analyticsObj, dataobj_ner_path):
        self.tagger = KeyphraseTagger(dataobj_ner_path)
        self.parser = spacy.load(nlpModel) if type(nlpModel) != spacy.lang.en.English else nlpModel
        self.analyticsObj = analyticsObj        
        
    def parseText(self, paragraph):
        paragraph = re.sub(r'\bid\b', 'identifier', paragraph)  # Spacy parses "id" as "i would", so fix here...
        doc = self.parser(paragraph)
        epm.mergeExcludePhrases(doc, self.parser.vocab)
        npm.mergeNounPhrasesDoc(doc, self.parser.vocab)
        return doc

    def extractData(self, depGraph, sentence, verbose=False):
        def isVerbNearestAncestor(targetVerb, exceptVerb):
            if targetVerb == exceptVerb:
                return True
            if exceptVerb.pos == spacy.symbols.VERB and self.tagger.getTag(exceptVerb).isShareOrCollect:
                return False
            if exceptVerb.head == exceptVerb:  # Hit the root
                return False
            return isVerbNearestAncestor(targetVerb, exceptVerb.head)

        def getReleventExceptions(verb, exceptions):
            if exceptions is None or len(exceptions) == 0:
                return exceptions
            return [(v, e) for v, e in exceptions if isVerbNearestAncestor(verb, v)]

        def getNearestAnnotVerb(depGraph, node):
            for s, _ in depGraph.in_edges(node):
                if s[2].isShareOrCollect:
                    return s

            for s, _ in depGraph.in_edges(node):
                res = getNearestAnnotVerb(depGraph, s)
                if res is not None:
                    return res
            return None

        def hasSubjectAndDobj(depGraph, node):
            hasSubject = any(n for _, n in depGraph.out_edges(node) if n[1].dep in [spacy.symbols.nsubj, spacy.symbols.nsubjpass])
            hasObject = any(n for _, n in depGraph.out_edges(node) if n[1].dep in [spacy.symbols.dobj])
            return hasSubject and hasObject

        def extractDataObjects(depGraph, baseNode):
            def extractDataObjectsInternal(results, depGraph, baseNode):
                for _, node in depGraph.out_edges(baseNode):
                    if node[2].isData:
                        results.append(node)
                    elif node[2].isShareOrCollect:  # Extract from NEAREST verb only
                        continue
                    elif node[1].pos == spacy.symbols.ADP and node[1].lemma_ in ['except when', 'except where', 'unless when', 'unless where', 'except for', 'except in', 'except under', 'unless for', 'unless in', 'unless under', 'apart from', 'aside from', 'with the exception of', 'other than', 'except to', 'unless to', 'unless as', 'except as']:
                        continue
                    extractDataObjectsInternal(results, depGraph, node)
            ##########################
            dataObjects = []
            # TODO if relcl, should we check the parent first?
            extractDataObjectsInternal(dataObjects, depGraph, baseNode)

            # Only do this if we don't have a direct object AND subject...
            if len(dataObjects) == 0 and not hasSubjectAndDobj(depGraph, baseNode):
                # Get from nearest parent?
                v = getNearestAnnotVerb(depGraph, baseNode)
                extractDataObjectsInternal(dataObjects, depGraph, v)
            return dataObjects

        def getAgent(depGraph, baseNode):
            def getEntityConjunctions(depGraph, node):
                def getEntityConjunctionsInternal(depGraph, node, res):
                    for _, target in depGraph.out_edges(node):
                        if depGraph[node][target]['label'] == 'conj':
                            res.append(target)
                            getEntityConjunctionsInternal(depGraph, target, res)
                    return res
                res = [node]
                res = getEntityConjunctionsInternal(depGraph, node, res)
                return res

            def getAgentInternal(depGraph, baseNode, skipTraverseUpwards=False, isXcomp=False):
                nsubj = None
                nsubjpass = None
                agentPobj = None
                dobj = None
                # Check children for the subject or agent if subject is passive...
                for _, node in depGraph.out_edges(baseNode):
                    if depGraph[baseNode][node]['label'] == 'nsubj':
                        nsubj = node
                    elif depGraph[baseNode][node]['label'] == 'nsubjpass':
                        nsubjpass = node
                    elif depGraph[baseNode][node]['label'] == 'dobj' or depGraph[baseNode][node]['label'] == 'dative':
                        dobj = node
                    elif depGraph[baseNode][node]['label'] == 'agent':  # "Agent" dependency tag
                        for _, node2 in depGraph.out_edges(node):
                            if node2[2].isEntity:
                                agentPobj = node2

                if nsubj is None:
                    nsubj = nsubjpass

                if isXcomp:
                    # If xcomp prefer dobj over nsubj...
                    if dobj is not None and dobj[2].isEntity:
                        return getEntityConjunctions(depGraph, dobj)
                    if nsubj is not None and nsubj[2].isEntity:
                        return getEntityConjunctions(depGraph, nsubj)
                    if nsubjpass is not None and nsubjpass[2].isEntity:
                        return getEntityConjunctions(depGraph, nsubjpass)
                    if agentPobj is not None and agentPobj[2].isEntity:
                        return getEntityConjunctions(depGraph, agentPobj)
                else:
                    if nsubj is not None and nsubj[2].isEntity:
                        return getEntityConjunctions(depGraph, nsubj)
                    if nsubjpass is not None and nsubjpass[2].isEntity:
                        return getEntityConjunctions(depGraph, nsubjpass)
                    if agentPobj is not None and agentPobj[2].isEntity:
                        return getEntityConjunctions(depGraph, agentPobj)
                    if dobj is not None and dobj[2].isEntity:
                        return getEntityConjunctions(depGraph, dobj)

                if not skipTraverseUpwards:
                    # If we don't find anything, get the parent verb if exists and search there
                    for node, _ in depGraph.in_edges(baseNode):
                        res = getAgentInternal(depGraph, node, skipTraverseUpwards=True, isXcomp=baseNode[1].dep in [spacy.symbols.xcomp, spacy.symbols.advcl])
                        if res is not None:
                            return res
                return None
            ##################
            agent = getAgentInternal(depGraph, baseNode)
            if agent is None or len(agent) == 0:  # If we haven't found anything return the default (i.e., "we") -- Rationale: "Personal information may be collected." means implicit "we"
                return ['we_implicit']  # Implicit first party
            return agent

        def ignoreActionObjectPair(verb, dobjects):
            if verb.lemma_ == 'send':  # Ignore send email or message
                for d in dobjects:
                    if re.search(r'.*\b(email|message)\b.*', d):
                        return True
            return False

        def getVerbNegationDistance(token, sentence):
            def isVerbNegatedInternal(token):
                for t in token.children:
                    if t.dep == spacy.symbols.neg:
                        # TODO need to record this somewhere for analytics purposes...
                        return t.i
                return -1
                # return any(t.dep == spacy.symbols.neg for t in token.children)

            dist = isVerbNegatedInternal(token)
            if dist >= 0:
                return dist

            # Check if verb is part of conjugated verb phrase, if so, check if any of those are negated
            conjugatedVerbs = DependencyGraphConstructor.getConjugatedVerbs(sentence, token)
            for tok in conjugatedVerbs:
                dist = isVerbNegatedInternal(tok)
                if dist >= 0:
                    return dist

            # Check if verb is xcomp, if so check if prior verb is negated? adjks
            if token.dep == spacy.symbols.xcomp:
                return getVerbNegationDistance(token.head, sentence)
            return -1

        def extractEntities(depGraph, baseNode):
            def extractEntitiesInternal(results, depGraph, baseNode):
                agent = getAgent(depGraph, baseNode)
                for _, node in depGraph.out_edges(baseNode):
                    if node[2].isEntity and node not in agent:
                        results.append(node)
                    elif node[2].isShareOrCollect:  # Extract from NEAREST annotated verb only
                        continue
                    elif node[1].pos == spacy.symbols.ADP and node[1].lemma_ in ['except when', 'except where', 'unless when', 'unless where', 'except for', 'except in', 'except under', 'unless for', 'unless in', 'unless under', 'apart from', 'aside from', 'with the exception of', 'other than', 'except to', 'unless to', 'unless as', 'except as']:
                        continue
                    extractEntitiesInternal(results, depGraph, node)
            ##########################
            entities = []
            extractEntitiesInternal(entities, depGraph, baseNode)
            return entities
        #########################

        def convertAgentToText(depGraph, agent):
            if agent is None:
                return agent
            result = []
            for a in agent:
                if type(a) == str:
                    result.append(a)
                    continue
                result.extend(depGraph.nodes[a]['lemmaList'])
            return result

        results = []
        if verbose:
            print('Found match.\n\t', sentence)

        # Start at the verbs...
        exceptions = eh.checkException(sentence)  # TODO should probably check the verb match here instead of doing it below...

        for n in depGraph:
            if n[2].isShareOrCollect:  # YZ: should we keep this condition?
                # dataObjects = [ d[1].lemma_ for d in extractDataObjects(depGraph, n) ]
                dataObjects = []
                for d in extractDataObjects(depGraph, n):
                    dataObjects.extend(depGraph.nodes[d]['lemmaList'])

                # entities = [ e[1].lemma_ for e in extractEntities(depGraph, n) ]
                entities = []
                for e in extractEntities(depGraph, n):
                    entities.extend(depGraph.nodes[e]['lemmaList'])

                agent = getAgent(depGraph, n)
                # Agent to text
                agent = convertAgentToText(depGraph, agent)

                if len(dataObjects) == 0 or ignoreActionObjectPair(n[1], dataObjects):  # skip <VERB, send>, <email>
                    continue

                actionLemmas = depGraph.nodes[n]['lemmaList']

                # Get related exceptions rooted under the specific share/collect verb...
                relExcepts = getReleventExceptions(n[1], exceptions)
                if verbose:
                    print(n, ('NOT', n[1].i, getVerbNegationDistance(n[1], sentence)) if depGraph.nodes[n]['neg'] else '')
                    print('\tDATA: ', dataObjects)
                    print('\tAGENT: ', agent)
                    print('\tENTITIES: ', entities)
                    # print '\tTYPE: ', ptype
                    print('\tEXCEPTIONS: ', exceptions)

                negDist = getVerbNegationDistance(n[1], sentence) if depGraph.nodes[n]['neg'] else -1
                results.append({'data_objects': dataObjects, 'entities': entities, 'agent': agent, 'action': n, 'action_lemmas': actionLemmas, 'is_negated': depGraph.nodes[n]['neg'], 'negation_distance': negDist, 'original_sentence': sentence.text, 'exceptions': relExcepts})
        return results

    def mix_extract(self, paragraph):

        results = []
        doc = self.parseText(paragraph)
        for sentence in doc.sents:
            tags = self.tagger.tagSentence(sentence)

            if len(tags) <= 0:
                continue

            # Skip interrogative sentences (Hao Cui)
            if sentence[-1].lemma_ == '?':
                continue

            # Prune the tree..
            depGraph = DependencyGraphConstructor.getSimplifiedDependencyGraph(sentence, tags)
            if len(tags) <= 0 or depGraph is None:
                continue

            res = self.extractData(depGraph, sentence)
            res = PolicyTransformer.handleExceptions(res, tags)
            res = PolicyTransformer.applyPolicyTransformationRules(res, self.analyticsObj)
            for r in res:
                results.append(r)

        return results if len(results) > 0 else None


def aggregateBySentence(policies):
    results = {}
    if policies is not None:  # We can just do extend instead of append if we're not going to be verbose here...
        for actor, collect, data, orig_sentence, actionLemma in policies:
            if orig_sentence not in results:
                results[orig_sentence] = set()
            results[orig_sentence].add((actor, collect, data, actionLemma))
    return results


def prettyPrintResults(policies):
    res = aggregateBySentence(policies)
    for sen in res:
        print(sen)
        for pol in res[sen]:
            print('\t', pol)

def val(v):
    res = v if type(v) == str else v.lemma_
    return res

def valTxt(v):
    res = v if type(v) == str else v.text
    return res

def dumpTree(tok, tab=''):
    print(tab, tok.lemma_, tok.pos_, tok.dep_, tok.i, tok.ent_type_)
    for child in tok.children:
        dumpTree(child, tab + '\t')

other_data = Path(r'D:\PPAudit\other_data')
output = Path(r'D:\PPAudit\output')
NlpFinalModel_Policheck = other_data / 'NlpFinalModel_Policheck'
pp_components = output / 'pp_components'
cus_phrase_tuples = output / 'cus_phrase_tuples'
AnalyticsData = output / 'analytics'
dataobj_ner_path = Path(r'D:\PPAudit\cus_extract\dataobj_ner')

def cus_phrse_tuple_extraction():
    nlp = spacy.load(NlpFinalModel_Policheck)
    analytics = Analytics()
    mix_extractor = MixExtractor(nlp, analytics)
    
    i = 0

    for pp_component_file in pp_components.glob('*.json'):
        i += 1
        pp_url_hash = pp_component_file.stem

        if AnalyticsData / ('{}.pickle'.format(pp_url_hash)).is_file() and \
        cus_phrase_tuples / ('{}.json'.format(pp_url_hash)).is_file():
            print('[ ] {}'.format(pp_url_hash))
            continue
        
        print('[+] {}'.format(pp_url_hash))

        with open(pp_component_file, 'r') as rf:  # YZ: to improve efficiency, only attempting to extract cus from cus sentences candidates
            file_content = json.loads(rf.read())
        cus_sentences_candidates = file_content[pp_url_hash]['CUS']

        analytics.startDoc(pp_url_hash)
        cus_phrases_result = {pp_url_hash: []}
        for line in cus_sentences_candidates:
            # print(line)
            res = mix_extractor.mix_extract(line)
            if res is not None:  # We can just do extend instead of append if we're not going to be verbose here...
                res = [(val(ent), val(col), val(dat), valTxt(sen), val(actionLemma)) for ent, col, dat, sen, actionLemma in res]
                for ent, col, dat, sen, actionLemma in res:
                    cus_phrases_result[pp_url_hash].append({
                        'sentence': sen,
                        'entity_phrase': ent,
                        'cus_or_not': col,
                        'dataobj_phrase': dat,
                        'cus_verb': actionLemma
                    })
                prettyPrintResults(res)
        
        # save cus phrase tuple results
        with open(cus_phrase_tuples/'{}.json'.format(pp_url_hash), 'w') as wf:
            wf.write(json.dumps(cus_phrases_result))

        # save analytics data
        analytics.endDoc()
        pickle.dump(analytics.dataStore[pp_url_hash], 
                    open(AnalyticsData/'{}.pickle'.format(pp_url_hash), 'wb'))
        
        if i == 5:
            break


if __name__ == '__main__':
    start_time = time.time()
    cus_phrse_tuple_extraction()
    end_time = time.time()
    print("[LOG]: run cus phrase tuple extraction cost {} s".format(end_time-start_time))