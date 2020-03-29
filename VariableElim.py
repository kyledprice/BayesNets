from collections import OrderedDict as ordered_dict
from BayesNet import BayesNet


class VariableElim(BayesNet):
    '''
        Represents a Baye's Net structure that is solved using Variable Elimination. Provide
        'parents' as an OrderedDict where the nodes point to a list of tuples containing their
        parents.

        Also provide the conditional probabilities at each node given their parent's probabilities.
        If a node doesn't have parents, it's just its sole probability.

        A custom printer can be provided to many functions that will print the results of the
        function call.

        An example instantiation could look like the following:

           parents = ordered_dict({'A': (), 'B': ('A'), 'C': ('A'), 'D': ('B'), 'E': ('B', 'C'), 'F': ('C')})
           probs = ordered_dict({'A':    {'+a': .3, '-a': .7},
                                 'B|A':  {'+a+b': .7, '+a-b': .3, '-a+b': .6, '-a-b': .4},
                                 'C|A':  {'+a+c': .2, '+a-c': .8, '-a+c': .9, '-a-c': .1},
                                 'D|B':  {'+b+d': .3, '+b-d': .7, '-b+d': .4, '-b-d': .6},
                                 'E|BC': {'+b+c+e': .2, '+b+c-e': .8,
                                          '+b-c+e': .6, '+b-c-e': .4,
                                          '-b+c+e': .1, '-b+c-e': .9,
                                          '-b-c+e': .5, '-b-c-e': .5},
                                 'F|C':  {'+c+f': .1, '+c-f': .9, '-c+f': .8, '-c-f': .2}})

           bn = VariableElim(parents, probs)

        The above example Baye's Net would have the structure:

                                        A
                                       /\
                                     B   C
                                    /\  /\
                                   D  E   F
    '''
    def __init__(self, parents, probs):
        super().__init__(parents)
        self.nodes = list(parents.keys())
        self.parents = parents
        self.num_vars = len(self.nodes)
        self.probs = probs

    '''
        Solve for the provided variables in the tuple 'values'. For now, all variables
        but 1 must have a hard value provided (e.g., +b or -c). The variable that can vary
        is just specified as None:

            ex: (None, '+e', '-f') assumes the jpt has been condensed to 3 variables and will
                allow the first variable to vary but restrict 'E' to '+e' and 'F' to '-f'.
    '''
    @classmethod
    def solve(cls, p_x, values, printer=None):
        pass

if __name__ == '__main__':
    parents = ordered_dict({'A': (), 'B': ('A'), 'C': ('A'), 'D': ('B'), 'E': ('B', 'C'), 'F': ('C')})
    probs = ordered_dict({'A':    {'+a': .3, '-a': .7},
                          'B|A':  {'+a+b': .7, '+a-b': .3, '-a+b': .6, '-a-b': .4},
                          'C|A':  {'+a+c': .2, '+a-c': .8, '-a+c': .9, '-a-c': .1},
                          'D|B':  {'+b+d': .3, '+b-d': .7, '-b+d': .4, '-b-d': .6},
                          'E|BC': {'+b+c+e': .2, '+b+c-e': .8,
                                   '+b-c+e': .6, '+b-c-e': .4,
                                   '-b+c+e': .1, '-b+c-e': .9,
                                   '-b-c+e': .5, '-b-c-e': .5},
                          'F|C':  {'+c+f': .1, '+c-f': .9, '-c+f': .8, '-c-f': .2}})
    bn = VariableElim(parents, probs)

