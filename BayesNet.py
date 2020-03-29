class BayesNet:
    '''
        Parent class that represents a Baye's Nets.
    '''
    def __init__(self, parents):
        self.nodes = list(parents.keys())
        self.num_vars = len(self.nodes)

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
    def sum_out_many(self, table, to_sum_out, printer=None):
        indices = [self.nodes.index(var) for var in to_sum_out]
        summed_out = table
        for j in range(len(indices)):
            summed_out = self.sum_out(summed_out, indices[j], printer)
            for k in range(j + 1, len(indices)):
                if indices[k] > indices[j]:
                    indices[k] -= 1
        return summed_out

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
