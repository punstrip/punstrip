import gensim
import dill
import IPython

with open('/root/desyl/res/corpus_sentences_skip_window_1.dill', 'rb') as fhandler:
    corpus_sentences = dill.load(fhandler)
    #sentences is a dict, need a flat list of sentences

    sentences = []
    for k, v in corpus_sentences.items():
        assert(isinstance(v, list))
        sentences += v

    IPython.embed()

    # build vocabulary and train model
    model = gensim.models.Word2Vec(
        sentences,
        size=150,
        window=2,
        min_count=2,
        workers=32)

    model.train(sentences, total_examples=len(sentences), epochs=100)
    IPython.embed()
