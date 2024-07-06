import networkx as nx
import pygraphviz as pgv
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import from_agraph
from ltlf2dfa.parser.ltlf import LTLfParser
from IPython.display import Image
import graphviz
from model.DFA import *

class LTL_Spec:
    def __init__(self, spec):
        self.spec = spec
        self.dfa = self.translate()

    def translate(self):
        parser = LTLfParser()
        formula = parser(self.spec)  # returns an LTLfFormula
        dfa_dot = formula.to_dfa()
        dfa_graph = self.to_network(dfa_dot)

        edges_info = {edge: dfa_graph.edges[edge] for edge in dfa_graph.edges}
        state_set = [*range(1, len(dfa_graph.nodes))]
        alphabet_set = []
        transitions_set = {}
        sink_states = []
        initial_state = 'init'
        for edge, _obs in edges_info.items():
            if _obs.get('label') is not None:
                obs = _obs.get('label')
                alphabet_set.append(obs)
                transitions_set[(edge[0], obs)] = edge[1]
            if (_obs.get('label') == "true") and (edge[0] == edge[1]):
                sink_states.append(edge[0])
            if (edge[0] == 'init') and (_obs.get('label') is None):
                initial_state = edge[1]
        dfa = DFA(states=state_set,
                  alphabet= alphabet_set,
                  transitions=transitions_set,
                  initial_state = initial_state,
                  sink_states= sink_states)
        print(dfa.transitions)
        print(dfa.initial_state)
        print(dfa.states)
        print(dfa.alphabet)
        return dfa


    def to_network(self, dfa_dot, plot_flag=False, image_flag=False):
        agraph = pgv.AGraph(string=dfa_dot)
        dfa_graph = from_agraph(agraph)
        if plot_flag:
            pos = nx.spring_layout(dfa_graph)
            nx.draw(dfa_graph, pos, with_labels=True)
            plt.show()
        if image_flag:
            _graph = graphviz.Source(dfa_dot)
            output_filename = 'MONA_DFA'
            _graph.render(output_filename, format='png', cleanup=True)
            Image(filename=f'{output_filename}.png')
        return dfa_graph

    # def condition(self, word_1, word_2):
    #     return f"({proposition_1} & {proposition_2})"



if __name__ == "__main__":

    # The syntax of LTLf is defined in the link: http://ltlf2dfa.diag.uniroma1.it/ltlf_syntax
    safe_frag = LTL_Spec("G(!o) & G(!g->!c)")
    # safe_frag = LTL_Spec("G(!b)")
    # scltl_frag = LTL_Spec("F(a)")


