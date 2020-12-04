import sys
import context
from tqdm import tqdm
import numpy as np
from classes.config import Config
from classes.binary import Binary
from classes.exporter import BinaryModifier
from classes.experiment import Experiment
from classes.callgraph import Callgraph
import classes.utils
import IPython

def infer_symbols_from_fp(b:Binary, exp:Experiment, clf_pipeline):
    ##pass in an analysed binary
    known_text_funcs    = set(map(lambda x: x.name, filter(lambda x: x.name != "func.{}".format(x.vaddr), b.symbols)))
    known_imported_funcs= set(map(lambda x: x['name'], b.dyn_imports))

    #calculate know functions
    known_functions = known_text_funcs | known_imported_funcs

    for i, s in tqdm(enumerate(b.symbols), desc='Inferring function names'):
        #skip knowns
        if s.name in known_text_funcs:
            continue

        symbol_vector   = s.to_vec(exp, KNOWN_FUNCS=known_functions)
        fp              = clf_pipeline(symbol_vector)
        inf_name        = exp.name_vector[ np.argmax(fp) ]

        print("{}   ->  {}".format(s.name, inf_name))
        b.symbols[i].name = inf_name

if __name__ == '__main__':
    """
        Unstrip a binary using learned models
    """
    path    = sys.argv[1]
    config  = Config()
    exp     = Experiment(config)
    exp.load_settings()
    b       = Binary(config, path=path)


    knc_clf         = classes.utils.pickle_load_py_obj(config, "knc_clf")
    pca             = classes.utils.pickle_load_py_obj(config, "pca")
    clf_pipeline    = lambda x: classes.callgraph.Callgraph.clf_proba_inf(knc_clf, exp, pca.transform(x))

    b.analyse()
    b.analyse_identify_static()

    infer_symbols_from_fp(b, exp, clf_pipeline)

    BM = BinaryModifier(config, b.path)
    BM.add_symbols(b.symbols)

    out_path = '{}.unstripped'.format(path)
    print("[+] Writing unstripped binary to {}".format(out_path))
    BM.save(out_path)


