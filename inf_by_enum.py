nodes = ['A', 'B', 'C', 'D', 'E', 'F']
p_A = {'+a': .3, '-a': .7}
p_BgA = {'+a+b': .7, '+a-b': .3, '-a+b': .6, '-a-b': .4}
p_CgA = {'+a+c': .2, '+a-c': .8, '-a+c': .9, '-a-c': .1}
p_DgB = {'+b+d': .3, '+b-d': .7, '-b+d': .4, '-b-d': .6}
p_EgC = {'+c+e': .4, '+c-e': .6, '-c+e': .5, '-c-e': .5}
p_FgC = {'+c+f': .1, '+c-f': .9, '-c+f': .8, '-c-f': .2}


# based on iteration number, return a tuple of of '+a' or '-a', '+b' or '-b', etc.
def sign(i):
    return ('+' + nodes[j].lower() if i & 2**exp < 1 else '-' + nodes[j].lower() for j, exp in enumerate(range(len(nodes) - 1, -1, -1), start=0))


def normalize(p_x):

    # get indices of variables that aren't given a specific value to solve for.
    # we need this because when we normalize, we have to normalize on all of them.
    # sums = {i: 0 for i, value in enumerate(values) if value is None}
    s = sum([row[-1] for row in p_x])
    for row in p_x:
        row[-1] /= s
    return p_x


def print_table(table):
    s = 0
    for i, row in enumerate(table, start=0):
        print(i, row)
        s += row[-1]
    print('sum = {}\n'.format(s))


def latex_print(table):
    for row in table:
        new_row = row.copy()
        new_row[-1] = str(round(new_row[-1], 5))
        print(' & '.join(new_row) + ' \\\\\n\\hline')
    print('\n')


# sum out the kth column in joint probability table
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


# sum out all of the variables listed in to_sum_out
def sum_out_many(p_x, to_sum_out, printer=None):
    indices = [nodes.index(var) for var in to_sum_out]
    p_j_summed_out = p_x
    for j in range(len(indices)):
        p_j_summed_out = sum_out(p_j_summed_out, indices[j], printer)
        for k in range(j + 1, len(indices)):
            if indices[k] > indices[j]:
                indices[k] -= 1
    return p_j_summed_out


# None -> no specific value
# +/-x -> indicates + or - for x
def solve(p_x, values):
    rows_to_remove = []
    for i, row in enumerate(p_x, start=0):
        matches = True
        for j, label in enumerate(row[0:-1], start=0):
            if values[j] is not None and values[j] != label:
                matches = False
                break
        if not matches:
            rows_to_remove.append(i)

    return normalize([val for i, val in enumerate(p_x, start=0) if i not in rows_to_remove])


# build full joint probability table
p_all = []
for i in range(2**6):
    a, b, c, d, e, f = sign(i)
    p_all.append([a, b, c, d, e, f, p_A[a] * p_BgA[a + b] * p_CgA[a + c] * p_DgB[b + d] * p_EgC[c + e] * p_FgC[c + f]])

print_table(p_all)
p_aef = sum_out_many(p_all, ['B', 'C', 'D'], printer=print_table)
final = solve(p_aef, (None, '+e', '-f'))
print_table(final)
