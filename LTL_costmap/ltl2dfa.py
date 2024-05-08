import networkx as nx
import pygraphviz as pgv
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import from_agraph
from ltlf2dfa.parser.ltlf import LTLfParser
from IPython.display import Image
import graphviz

class DFA:
    def __init__(self):
        self.parser = LTLfParser()

    def translate(self, formula_str):
        formula = self.parser(formula_str)  # returns an LTLfFormula
        dfa = formula.to_dfa()
        return dfa

    def to_network(self, dfa_dot):
        agraph = pgv.AGraph(string=dfa_dot)
        graph = from_agraph(agraph)
        pos = nx.spring_layout(graph)
        return graph
        # nx.draw(graph, pos, with_labels=True)
        # plt.show()


    def display(self, dfa):
        graph = graphviz.Source(dfa)
        output_filename = 'MONA_DFA'
        graph.render(output_filename, format='png', cleanup=True)
        Image(filename=f'{output_filename}.png')




if __name__ == "__main__":
    ltl2dfa = DFA()
    spec = "G(a -> X b)"
    dfa = ltl2dfa.translate(spec)
    ltl2dfa.to_network(dfa)
    print(dfa)

