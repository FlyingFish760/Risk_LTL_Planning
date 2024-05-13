import networkx as nx
import pygraphviz as pgv
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import from_agraph
from ltlf2dfa.parser.ltlf import LTLfParser
from IPython.display import Image
import graphviz
from model.DFA import *

class LTL2DFA:
    def __init__(self, spec):
        self.spec = spec
        self.dfa_graph = None
        self.dfa = self.translate()

    def translate(self):
        parser = LTLfParser()
        formula = parser(self.spec)  # returns an LTLfFormula
        dfa_dot = formula.to_dfa()
        self.to_network(dfa_dot)

        edges_info = {edge: self.dfa_graph.edges[edge] for edge in self.dfa_graph.edges}
        alphabet_set = []
        transitions_set = {}
        for edge, _obs in edges_info.items():
            obs = _obs.get('label') if _obs.get('label') is not None else ' '
            alphabet_set.append(obs)
            transitions_set[(edge[0], obs)] = edge[1]
        print(transitions_set)
        print(alphabet_set)
        dfa = DFA(states=list(self.dfa_graph.nodes),
                  alphabet= alphabet_set,
                  transitions=transitions_set,
                  initial_state = 'init',
                  accept_states= '3')
        return dfa


    def to_network(self, dfa_dot, plot_flag=False, image_flag=False):
        agraph = pgv.AGraph(string=dfa_dot)
        self.dfa_graph = from_agraph(agraph)
        if plot_flag:
            pos = nx.spring_layout(self.dfa_graph)
            nx.draw(self.dfa_graph, pos, with_labels=True)
            plt.show()
        if image_flag:
            _graph = graphviz.Source(dfa_dot)
            output_filename = 'MONA_DFA'
            _graph.render(output_filename, format='png', cleanup=True)
            Image(filename=f'{output_filename}.png')



if __name__ == "__main__":
    spec = "G(a -> X b)"
    ltl2dfa = LTL2DFA(spec)


