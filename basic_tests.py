from InfByEnum import InfByEnum
from BayesNet import BayesNet
from VariableElim import VariableElim
from collections import OrderedDict as ordered_dict

if __name__ == '__main__':
    parents = ordered_dict({'C': (), 'D': ('C', 'M', 'T'), 'M': ('W'), 'T': ('W'), 'W': ()})
    probs = ordered_dict({'C':     {'+c': .5, '-c': .5},
                          'D|CMT': {'+c+d+m+t': .9, '+c+d+m-t': .8, '+c+d-m+t': .8, '+c+d-m-t': .2,
                                    '+c-d+m+t': .1, '+c-d+m-t': .2, '+c-d-m+t': .2, '+c-d-m-t': .8,
                                    '-c+d+m+t': .8, '-c+d+m-t': .5, '-c+d-m+t': .6, '-c+d-m-t': .1,
                                    '-c-d+m+t': .2, '-c-d+m-t': .5, '-c-d-m+t': .4, '-c-d-m-t': .9},
                          'M|W':   {'+m+w': .8, '+m-w': .1, '-m+w': .2, '-m-w': .9},
                          'T|W':   {'+t+w': .7, '+t-w': .9, '-t+w': .3, '-t-w': .1},
                          'W':     {'+w': .9, '-w': .1}})
    bn = InfByEnum(parents, probs)

    p_aef = bn.sum_out_many(bn.jpt, ['C', 'T'])
    inf_result = bn.solve(p_aef, ('+d', '+m', '+w'))
    BayesNet.print_table(inf_result)

    bn = VariableElim(parents, probs)
    ve_result = bn.solve((None, '+d', '+m', None, '+w'))
    BayesNet.print_table(ve_result)

    assert ve_result == inf_result
