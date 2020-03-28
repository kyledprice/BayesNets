from collections import OrderedDict as ordered_dict


class BayesNet:
    '''
        Represents a Baye's Net structure. Provide 'parents' as an OrderedDict where
        the nodes point to a list of tuples containing their parents.

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

           bn = BayesNet(parents, probs)

           # P(A | +e, -f)
           p_aef = bn.sum_out_many(['B', 'C', 'D'])
           final = bn.solve(p_aef, (None, '+e', '-f'), printer=BayesNet.print_table)

           # P(+e, -b, +c, -d, +e, -f)
           final = bn.solve(bn.jpt, ('+a', '-b', '+c', '-d', '+e', '-f'), printer=BayesNet.print_table)

       The above example Baye's Net would have the structure:

                                        A
                                       /\
                                     B   C
                                    /\  /\
                                   D  E   F
    '''
    def __init__(self, parents, probs, printer=None):
        self.nodes = list(parents.keys())
        self.parents = parents
        self.num_vars = len(self.nodes)
        self.probs = probs
        self.jpt = self.build_jpt(printer)

    ''' 
        Build full joint probability table. Format for jpt with variables 
        A and B matches the following:
                            
                                ['+a, '+b', prob]
                                ['+a, '-b', prob]
                                ['-a, '+b', prob]
                                ['-a, '-b', prob]
    '''
    def build_jpt(self, printer):
        jpt = []
        for i in range(2**self.num_vars):
            var_signs = self.signs(i)
            jpt_product = 1 # start at 1 so residual product can occur (0 *= ANYTHING => 0)
            for n, n_parents in self.parents.items():
                sign_indices = [self.nodes.index(var) for var in sorted([n] + list(n_parents))]
                prob_key = ''.join([var_signs[sign_idx] + self.nodes[sign_idx].lower() for sign_idx in sign_indices])
                prob_table_key = n if not len(n_parents) else '{}|{}'.format(n, ''.join(n_parents))
                jpt_product *= self.probs[prob_table_key][prob_key]
            var_signs_and_values = [var_signs[j] + var.lower() for j, var in enumerate(self.nodes, start=0)]
            jpt.append(var_signs_and_values + [jpt_product])
        if printer is not None:
            printer(jpt)
        return jpt

    ''' 
        Based on iteration number, return a list of of signs for each variable in self.nodes.
        Moreorless creates a row on a truth table. 
    '''
    def signs(self, i):
        var_signs = []
        for exp in range(self.num_vars - 1, -1, -1):
            var_sign = '+' if (i & 2**exp) < 1 else '-'
            var_signs.append(var_sign)
        return var_signs

    ''' 
        Sum out all of the variables listed in to_sum_out.
    '''
    def sum_out_many(self, to_sum_out, printer=None):
        indices = [self.nodes.index(var) for var in to_sum_out]
        summed_out = self.jpt
        for j in range(len(indices)):
            summed_out = self.sum_out(summed_out, indices[j], printer)
            for k in range(j + 1, len(indices)):
                if indices[k] > indices[j]:
                    indices[k] -= 1
        return summed_out

    ''' 
        Solve for the provided variables in the tuple 'values'. For now, all variables 
        but 1 must have a hard value provided (e.g., +b or -c). The variable that can vary 
        is just specified as None:
            
            ex: (None, '+e', '-f') assumes the jpt has been condensed to 3 variables and will 
                allow the first variable to vary but restrict 'E' to '+e' and 'F' to '-f'. 
    '''
    @classmethod
    def solve(cls, p_x, values, printer=None):

        # Find the rows that need to be removed.
        rows_to_remove = []
        for i, row in enumerate(p_x, start=0):
            matches = True
            for j, label in enumerate(row[0:-1], start=0):
                if values[j] is not None and values[j] != label:
                    matches = False
                    break
            if not matches:
                rows_to_remove.append(i)

        # Build table with rows that match the provided variables. Don't normalize them
        # if all variable values were specified.
        table = [val for i, val in enumerate(p_x, start=0) if i not in rows_to_remove]
        if len(table) != 1:
            table = cls.normalize(table)
        if printer is not None:
            printer(table)
        return table

    '''
        Ensure all row probabilities sum to 1.
    '''
    @staticmethod
    def normalize(p_x):
        s = sum([row[-1] for row in p_x])
        for row in p_x:
            row[-1] /= s
        return p_x

    @staticmethod
    def print_table(table):
        s = 0
        for i, row in enumerate(table, start=0):
            print(i, row[:-1] + [round(row[-1], 5)])
            s += row[-1]
        print('sum = {}\n'.format(s))

    @staticmethod
    def latex_print(table):
        for row in table:
            new_row = row.copy()
            new_row[-1] = str(round(new_row[-1], 5))
            print(' & '.join(new_row) + ' \\\\\n\\hline')
        print('\n')

    '''
        Sum out the kth column in joint probability table.
    '''
    @staticmethod
    def sum_out(p_x, k, printer=None):
        p_new = []
        collapsed = set()
        for i in range(len(p_x)):
            if i in collapsed:
                continue
            adjusted_idx = i + (len(p_x) >> 1 + k)
            s = p_x[i][-1] + p_x[adjusted_idx][-1]
            p_new.append(p_x[i][0:k] + p_x[i][k + 1:len(p_x[i]) - 1] + [s])
            collapsed.add(adjusted_idx)
        if printer is not None:
            printer(p_new)
        return p_new


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
    bn = BayesNet(parents, probs)

    # P(A | +e,-f)
    p_aef = bn.sum_out_many(['B', 'C', 'D'])
    bn.solve(p_aef, (None, '+e', '-f'), printer=BayesNet.print_table)

    # P(A,B | +e,-f) -> not sure this correct yet
    p_abef = bn.sum_out_many(['C', 'D'])
    bn.solve(p_abef, (None, None, '+e', '-f'), printer=BayesNet.print_table)

    # P(+e,-b,+c,-d,+e,-f)
    bn.solve(bn.jpt, ('+a', '-b', '+c', '-d', '+e', '-f'), printer=BayesNet.print_table)
