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
       Join and eliminate each hidden variable, then join the remaining tables. 
    '''
    def solve(self, values, printer=None):
        just_evidence_vars = [var for var in values if var is not None and not var.isupper()]
        tables = self.enact_evidence(just_evidence_vars, printer)
        hidden_vars = self.get_hidden_vars(values)
        for var in hidden_vars:
            tables, key = self.join_and_eliminate(var, tables, printer)
        joined = tables[key] if len(tables) == 1 else self.join(tables, list(tables.keys()))
        is_distribution = len(values) != len(hidden_vars) + len(just_evidence_vars)
        final = self.normalize(joined) if is_distribution else joined
        if printer:
            BayesNet.print_table(final)
        return self.round_last(final, 5)

    def enact_evidence(self, values, printer):

        # Create a dict where variable names of all variables whose values are provided as evidence point
        # to the index in values that has their sign.
        evidence_dict = {var.replace('+', '').replace('-', ''): i for i, var in enumerate(values, start=0)}

        tables = self.probs.copy()
        for prob_table_key, prob_dict in self.probs.items():
            table = []
            for prob_var_key, prob_value in prob_dict.items():
                cols = [prob_var_key[i:i + 2] for i in range(0, len(prob_var_key), 2)]

                # Make sure rows match evidence.
                all_evidence_matches = True
                for evidence_var, values_idx in evidence_dict.items():
                    if evidence_var in prob_var_key and values[values_idx] not in cols:
                        all_evidence_matches = False
                        break
                if all_evidence_matches:
                    table.append(cols + [prob_value])
            tables[prob_table_key] = table
        if printer:
            printer(tables)

        return tables

    def get_hidden_vars(self, values):
        nonhidden_vars = [var.upper().replace('+', '').replace('-', '') for var in values if var is not None]
        return [var for var in self.nodes if var not in nonhidden_vars]

    def join(self, tables, key_queue):
        var_expanded_table = tables[key_queue.pop(0)]
        while len(key_queue) > 0:
            next_table = tables[key_queue.pop(0)]
            temp_table = []
            for row in var_expanded_table:
                row_set = set(row[:-1])
                for other_row in next_table:
                    other_row_set = set(other_row[:-1])
                    if self.evidence_matches(row_set, other_row_set):
                        new_row = sorted(list(row_set.union(other_row_set)), key=lambda v: v[1])
                        temp_table.append(new_row + [row[-1] * other_row[-1]])
            var_expanded_table = temp_table
        return var_expanded_table

    def join_and_eliminate(self, var, tables, printer):
        keys_with_var = [table_key for table_key in tables.keys() if var in table_key]
        key_queue = keys_with_var.copy()
        table_key = '_'.join(key_queue)
        var_expanded_table = self.join(tables, key_queue)
        for key in keys_with_var:
            del tables[key]

        tables[table_key] = var_expanded_table

        for k, col in enumerate(var_expanded_table[0][:-1], start=0):
            if var.lower() in col:
                break

        tables[table_key] = VariableElim.sum_out(tables[table_key], k)
        if printer:
            printer(tables)
        return tables, table_key

    @staticmethod
    def evidence_matches(row_set, other_row_set):
        same = row_set.intersection(other_row_set)
        row_set_unique = set(var[1:] for var in row_set.difference(same))
        other_row_set_unique = set(var[1:] for var in other_row_set.difference(same))
        return len(same) != 0 and row_set_unique.isdisjoint(other_row_set_unique)

    @staticmethod
    def sum_out(p_x, k):
        p_new = []
        collapsed = set()
        for i in range(len(p_x)):
            vars_i = p_x[i][:k] + p_x[i][k + 1:-1]
            for j in range(i + 1, len(p_x)):
                if i in collapsed or j in collapsed:
                    continue
                vars_j = p_x[j][:k] + p_x[j][k + 1:-1]
                if vars_i == vars_j:
                    collapsed.add(j)
                    p_new.append(vars_i + [p_x[i][-1] + p_x[j][-1]])
        return p_new

    @staticmethod
    def print_tables(tables):
        for prob_table_key, prob_list in tables.items():
            s = 0
            print('Table: ' + prob_table_key)
            for i, row in enumerate(prob_list, start=0):
                print(i, row[:-1] + [round(row[-1], 5)])
                s += row[-1]
            print('sum = {}\n'.format(s))


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

    # P(A | +e,-f)
    bn.solve(('A', None, None, None, '+e', '-f'), printer=VariableElim.print_tables)

    # P(+e,-b,+c,-d,+e,-f)
    bn.solve(('+a', '-b', '+c', '-d', '+e', '-f'), printer=VariableElim.print_tables)
