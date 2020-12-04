import scipy
import subprocess
import IPython


def get_calculated_knowns(nlp, path):
    """
        Parse the list of dynamic symbol FUNC names.
        Functions such as libc_csu_init, fini, and start are in teh dynsym table but are in the .text section of the binary
    """
    cmd = "readelf -D -s {} | grep 'FUNC'".format(path) + "| awk '{print $9}'"
    dynsyms = subprocess.check_output(cmd, shell=True).decode('ascii').split('\n')
    calculated_knowns = set(['start', 'csu_fini', 'csu_init'])
    for dynsym in dynsyms:
        name = nlp.strip_library_decorations( dynsym )
        calculated_knowns.add(name)
    return calculated_knowns

def print_possible_permutations(node_sets):
    possible = '*'.join(list(map(lambda x: str(len(x)), node_sets.values())))
    print("Possible permutations: {}".format(possible))


def calculate_unknown_permutations(G, feature_functions, name_to_index, index_to_name):
    """
        G is a CRF with knowns and unknowns. For each known in the graph, compute the set of possible 
        symbols each unknown could take. Union each possible set of unknowns for each known. Then 
        calculate the number of possible permutations over Y as prod(len(s)) for s in S where S is a 
        list of sets of symbols each node in the crf can take.

        :param ffs: A list of feature functions [(forward, reverse), ...] connecting symbols -> symbols
    """
    assert(isinstance(feature_functions, list))
    knowns_in_bin = set(filter(lambda x: G.nodes[x]['func'] and not G.nodes[x]['text_func'], G.nodes()))
    unknowns_in_bin = set(filter(lambda x: G.nodes[x]['func'] and G.nodes[x]['text_func'], G.nodes()))

    #init sets of possible symbols for all unknowns
    symbols_sets = dict([node, set([])] for node in unknowns_in_bin)


    for ff, ffr in feature_functions:
        for node in knowns_in_bin:
            #print("Node:", node)
            for successor in G.successors(node):
                if not G[node][successor]['call_ref'] or sucessor in knowns_in_bin:
                    continue

                r, c = ff[name_to_index[node], :].nonzero()
                possible_symbols = set(map(lambda x: index_to_name[x], c))
                for possible_symbol in possible_symbols:
                    if possible_symbol not in symbols_sets[successor]:
                        symbols_sets[successor].add(possible_symbol)

            for predecessor in G.predecessors(node):
                #print("predecessor:", predecessor)
                #import IPython
                #IPython.embed()
                if not G[predecessor][node]['call_ref'] or predecessor in knowns_in_bin:
                    continue

                r, c = ffr[name_to_index[node], :].nonzero()
                possible_symbols = set(map(lambda x: index_to_name[x], c))
                for possible_symbol in possible_symbols:
                    if possible_symbol not in symbols_sets[predecessor]:
                        symbols_sets[predecessor].add(possible_symbol)

    print("NB: This is only after looking at direct calls etween knowns and unknowns and not recursivly unknowns -> unknowns. The real number is far bigger than this...")
    print_possible_permutations(symbols_sets)
    import IPython
    IPython.embed()
