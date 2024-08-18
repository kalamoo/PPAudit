import copy
import networkx as nx


class tupleGranularity:
    def __init__(self, data_onto_path, entity_onto_path):
        # data_onto = r"E:\PPAudit\E_Formalization\data\final-ontology-and-synonyms\data_ontology.gml"
        self.data_graph = nx.read_gml(data_onto_path)
        self.data_leaves = []
        for node in self.data_graph.nodes():
            if self.data_graph.out_degree[node] == 0:
                self.data_leaves.append(node)

        # entity_onto = r"E:\PPAudit\E_Formalization\data\final-ontology-and-synonyms\entity_ontology.gml"
        self.entity_graph = nx.read_gml(entity_onto_path)
        self.entity_leaves = []
        for node in self.entity_graph.nodes():
            if self.entity_graph.out_degree[node] == 0:
                self.entity_leaves.append(node)

        self.data_granularity = self.calc_depth_all(mode='data')
        self.entity_granularity = self.calc_depth_all(mode='entity')

    def calc_depth(self, node, mode='data'):
        #  max(max(simple_path_leaf()))）
        leaves = self.data_leaves if mode=='data' else self.entity_leaves
        graph = self.data_graph if mode=='data' else self.entity_graph
        granularity_path = []
        for leaf in leaves:
            simple_paths = nx.all_simple_paths(G=graph, source=node, target=leaf)
            for simple_path in simple_paths:
                if len(simple_path) > len(granularity_path):
                    granularity_path = simple_path
        return max(len(granularity_path), 1)
    
    def calc_depth_all(self, mode='data'):
        depth_nodes_dict = dict()
        graph = self.data_graph if mode=='data' else self.entity_graph
        for node in graph.nodes():
            depth = self.calc_depth(node, mode=mode)
            if depth not in depth_nodes_dict.keys():
                depth_nodes_dict[depth] = [node, ]
            else:
                depth_nodes_dict[depth].append(node)
        return depth_nodes_dict        


class appGranularity:
    def __init__(self, data_onto_path):
        self.graph = nx.read_gml(data_onto_path)

        self.leaves = []
        for node in self.graph.nodes():
            if self.graph.out_degree[node] == 0:
                self.leaves.append(node)
        
        self.depth_nodes_dict = self.calc_depth_all()

    def calc_depth(self, node):
        #  max(max(simple_path_leaf()))）
        granularity_path = []
        for leaf in self.leaves:
            simple_paths = nx.all_simple_paths(G=self.graph, source=node, target=leaf)
            for simple_path in simple_paths:
                if len(simple_path) > len(granularity_path):
                    granularity_path = simple_path
        return max(len(granularity_path), 1)
    
    def calc_depth_all(self):
        depth_nodes_dict = dict() # key: 1 ~ 5
        for node in self.graph.nodes():
            depth = self.calc_depth(node)
            if depth not in depth_nodes_dict.keys():
                depth_nodes_dict[depth] = [node, ]
            else:
                depth_nodes_dict[depth].append(node)
        return depth_nodes_dict

    def data_claim(self, cus_tuples):
        results = set([cus_tuple[2] for cus_tuple in cus_tuples])
        return results

    def data_lower_bound(self, cus_tuples):
        original_nodes = list(set([cus_tuple[2] for cus_tuple in cus_tuples]))
        
        # init with leaf nodes
        results = set(original_nodes).intersection(set(self.leaves))
        
        # iter nodes with depth ascending
        for depth in range(2, 6):
            tmp_results = copy.copy(results)
            nodes_in_depth = set(original_nodes).intersection(self.depth_nodes_dict[depth])
            delete_nodes = set() 
            for node_in_depth in nodes_in_depth:
                for node_in_result in tmp_results:
                    if nx.has_path(self.graph, node_in_depth, node_in_result):
                        delete_nodes.add(node_in_depth)
                        break
            results = results.union(nodes_in_depth).difference(delete_nodes)
        return results


    def data_upper_bound(self, cus_tuples):
        original_nodes = list(set([cus_tuple[2] for cus_tuple in cus_tuples]))

        results = set(original_nodes)
        for node in original_nodes:
            results = results.union(nx.descendants(self.graph, node))
        return results
