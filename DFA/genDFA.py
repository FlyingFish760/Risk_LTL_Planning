from ltlf2dfa.parser.ltlf import LTLfParser

parser = LTLfParser()
formula_str = "G(a -> X b)"
formula = parser(formula_str)       # returns an LTLfFormula
dfa = formula.to_dfa()
print(dfa)