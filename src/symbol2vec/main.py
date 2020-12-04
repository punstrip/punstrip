import os, sys
import tensorflow as tf
import numpy as np
import random, logging
import itertools, tqdm
import collections, functools
import math
import context
import classes.utils
import classes.callgraph
from classes.config import Config
from classes.database import Database
from sklearn.model_selection import train_test_split
from tensorflow.contrib.tensorboard.plugins import projector

stop_symbols = set([ 'stack_chk_fail', 'stack_chk_fail_local', '', 'errno_location' ])

def generate_epoch_sample(corpus, name_to_index, k=1):
	"""
	So that the distribution of I/Os is uniform we need to randomly permutate symbols per epoch.
	pivot_words is a list of a subpermutation of all symbols for this batch
	Pick batch_size random words and sample k items from each word weighted by the copus counters
	k is the number of samples from the context
	if a symbol doesn't appear in any contexts then it only appears by itself
	"""
	symbol_names = set(name_to_index.keys())
	#pivot_words = random.sample(corpus.keys(), batch_size)
	pivot_words = list(name_to_index.keys())
	random.shuffle(pivot_words)
	out_pivot_words, out_target_words = [], []

	for i in tqdm.tqdm(range(len(pivot_words))):
		context_counts = corpus[ pivot_words[i] ].items()

		#filter contexts to be in restricted name_to_index
		filtered_context_counts = list(filter(lambda x, symbol_names=symbol_names: x[0] in symbol_names, context_counts))

		if len(filtered_context_counts) == 0:
			print("WARING: SYMBOL ", pivot_words[i], "HAS NO TARGET WORDS AFTER FILTERING")
			continue

		context_symbols, weights = zip(*filtered_context_counts)
		target_words = random.choices(context_symbols, weights=weights, k=k)

		## create 1 IN/OUT
		out_target_words += target_words
		out_pivot_words += [ pivot_words[i] ] * k

	return out_pivot_words, out_target_words

def generate_epoch_all(corpus, name_to_index):
	"""
	Flatten corpus to single IO for all occourances in name_to_index
	"""
	symbol_names = set(name_to_index.keys())
	#pivot_words = random.sample(corpus.keys(), batch_size)
	pivot_words = list(name_to_index.keys())
	random.shuffle(pivot_words)
	out_pivot_words, out_target_words = [], []

	for i in tqdm.tqdm(range(len(pivot_words))):
		context_words = corpus[ pivot_words[i] ].elements()

		#filter contexts to be in restricted name_to_index
		filtered_context_words = list(filter(lambda x: x in symbol_names, context_words))

		k = len(filtered_context_words)
		if k == 0:
			continue

		## create 1 IN/OUT
		out_target_words += filtered_context_words
		out_pivot_words += [ pivot_words[i] ] * k

	return out_pivot_words, out_target_words

def sentence2vec(name_to_index, sentence):
	"""
	Convert a list of symbol names to a vector using one-hot encoding 
	"""
	N = len(name_to_index)
	vec = np.ndarray(shape=(N, 1), dtype=np.uint8)
	for word in sentence:
		ind = name_to_index[word]
		vec[ind, 0] = 1
	return vec

def _emit_nth_parents(G, node, path, n):
	predecessors  = list(filter(lambda x: G[x][node]['call_ref'], G.predecessors(node)))
	if n == 0 or len(predecessors) == 0:
		return list(path) + [node]
	return list(map(lambda x: _emit_nth_parents(G, x, list(path) + [node], n-1), predecessors)) 

def _emit_nth_children(G, node, path, n):
	#import IPython
	#IPython.embed()
	successors = list(filter(lambda x: G[node][x]['call_ref'], G.successors(node)))
	if n == 0 or len(list(successors)) == 0:
		return list(path) + [node]
	return list(map(lambda x: _emit_nth_children(G, x, list(path) + [node],  n-1), successors))

def _rec_unravel_sentences(s):
	assert(isinstance(s, list))
	sentences = []
	end_sentence = True
	for i in s:
		#we are not a sentence but a list of sentences
		if isinstance(i, list):
			sentences += _rec_unravel_sentences(i)
			end_sentence = False

	if end_sentence:
		sentences.append( s )

	return sentences

def emit_sentences_from_callgraphs(GG, symbol, skip_window=2):
	sentences = []

	for G in GG:
		if symbol not in G.nodes():
			continue

		symbol_sentences = []

		child_paths = _emit_nth_children(G, symbol, [], skip_window)
		parent_paths = _emit_nth_parents(G, symbol, [], skip_window)

		child_sentences = _rec_unravel_sentences(child_paths)
		parent_sentences = _rec_unravel_sentences(parent_paths)

		for s in child_sentences + parent_sentences:
			symbol_sentences.append(s)
		sentences += symbol_sentences

	return sentences

def emit_combinatorial_sentences_from_callgraphs(GG, symbol, skip_window=2):
	sentences = []

	for G in GG:
		if symbol not in G.nodes():
			continue

		symbol_sentences = []

		child_paths = _emit_nth_children(G, symbol, [], skip_window)
		parent_paths = _emit_nth_parents(G, symbol, [], skip_window)

		child_sentences = _rec_unravel_sentences(child_paths)
		parent_sentences = _rec_unravel_sentences(parent_paths)

		_parent_sentences = []

		for s in parent_sentences:
			#reverse and remove node "symbol". Then when parents + children are added, symbol is in children
			r = s[::-1]
			del r[-1]
			if len(r) > 0:
				_parent_sentences.append( r )

		#unique_sentences = itertools.product(child_sentences, _parent_sentences)
		#when taking product, if child || parent are empty we get no sentences. We want maximum length sentences so only append null sentence 
		#if empty
		if len(child_sentences) == 0:
			child_sentences.append([])
		if len(_parent_sentences) == 0:
			_parent_sentences.append([])

		for c in child_sentences:
			for p in _parent_sentences:
				assert(isinstance(c, list))
				assert(isinstance(p, list))
				s = p + c
				if len(s) > 1:
					symbol_sentences.append(s)

		#print(symbol_sentences)
		for s in symbol_sentences:
			max_len = 1 + (skip_window * 2)
			if(len(s) > max_len):
				raise Exception("ERROR: Length of sentence longer than {} - {}".format(max_len, s))

		sentences += symbol_sentences

	return sentences

def generate_symbol_sentences(GG, name_to_index, skip_window=2, combinations=False):
	"""
	Generate array of symbols contexts (i_-1, i_0, i_+1)
	"""
	corpus = {}
	for symbol in tqdm.tqdm(name_to_index.keys()):
		if symbol == '':
			continue

		if not combinations:
			sentences = emit_sentences_from_callgraphs(GG, symbol, skip_window)
		else:
			sentences = emit_combinatorial_sentences_from_callgraphs(GG, symbol, skip_window)
			
		#context_vectors = list(map(lambda x: sentence2vec(name_to_index, x), sentences))
		#corpus[symbol] = context_vectors
		corpus[symbol] = sentences

	return corpus

def generate_corpus_sampler(GG, name_to_index, stop_symbols, skip_window=2, num_skips=2):
	"""
	Generate dictonary of symbols contexts (i_-1, i_0, i_+1)
	Each symbol has a collections.Counter with the count of all symbols in the symbols context. 
	The corpus is then passed into random.choices to sample N symbls from the pivot words context
	"""
	#use a counter from collections. Then pass into random.choices and sample it
	corpus = {}
	for symbol in tqdm.tqdm(name_to_index.keys()):
		if symbol in stop_symbols:
			continue
		symbol_counter = collections.Counter()
		sentences = emit_sentences_from_callgraphs(GG, symbol, skip_window)
		for sentence in sentences:
			for word in sentence:
				if word in stop_symbols:
					continue
				symbol_counter[word] += 1

		#remove self references/loops to symbol in callgraph
		del symbol_counter[symbol]
		corpus[symbol] = symbol_counter
	return corpus

def _subsampling_p_wi(f_wi, t=10.0e-4):
	"""
		Compute the probability of droping word wi from 
			P(w_i) = 1 - sqrt( t / f(w_i) )
	"""
	return 1 - math.sqrt( t / float(f_wi) )


def subsample_corpus(corpus, lowest_freq=10, highest_freq=100000):
	"""
		Remove symbols with a low number of contexts and too high!
	"""
	frequencies = {}
	for k in tqdm.tqdm(corpus.keys()):
		count = functools.reduce(lambda x, y: x+y, corpus[k].values(), 0)
		frequencies[k] = count

	high_pass_freqs = dict(filter(lambda x: x[1] >= lowest_freq, frequencies.items() ))
	logger.info("corpus contains {} symbol names after being filtered with the lowest_frequency being {}".format(len(high_pass_freqs), lowest_freq))
	low_pass_freqs = dict(filter(lambda x: x[1] <= highest_freq, high_pass_freqs.items() ))
	logger.info("corpus contains {} symbol names after being filtered with the highest_frequency being {}".format(len(low_pass_freqs), highest_freq))

	"""
	Subsampling doesn't work
	probs = {}
	#convert freqs to probabilities of dropping
	for k, v in filtered_freqs.items():
		probs[k] = _subsampling_p_wi(v)
	import IPython
	IPython.embed()
	"""

	new_symbol_names = set(low_pass_freqs.keys())
	#generate new name_to_index, index_to_name and corpus
	new_corpus = dict(filter(lambda x: x[0] in new_symbol_names, corpus.items()))
	new_name_to_index, new_index_to_name = {}, {}
	index = 0
	for name in new_symbol_names:
		new_name_to_index[name]		= index
		new_index_to_name[index]	= name
		index += 1

	return new_corpus, new_name_to_index, new_index_to_name

def remove_symbols_with_no_valid_contexts(corpus, name_to_index):
	symbol_names = set(name_to_index.keys())
	corpus_names = set(corpus.keys())
	assert(len(symbol_names) == len(corpus_names))

	for name in tqdm.tqdm(corpus_names):
		context_counts = corpus[ name ].items()
 
		#filter contexts to be in restricted name_to_index
		filtered_context_counts = list(filter(lambda x, symbol_names=symbol_names: x[0] in symbol_names, context_counts))

		if len(filtered_context_counts) == 0:
			del corpus[name]

	new_symbol_names = set(corpus.keys())
	#generate new name_to_index, index_to_name and corpus
	new_name_to_index, new_index_to_name = {}, {}
	index = 0
	for name in new_symbol_names:
		new_name_to_index[name]		= index
		new_index_to_name[index]	= name
		index += 1

	return dict(corpus), new_name_to_index, new_index_to_name





def build_examples(name_to_index):
	arr = ['free', 'malloc', 'md5_init_ctx', 'sha512_init_ctx', 'md5_stream', 'sha512_read_ctx', 'sha512_stream', 'sha512_process_bytes', 'mutt_md5_init_ctx', 'main', 'exit', 'MD5_Init', 'SHA512_Init', 'MD5_Final', 'SHA512_Final']
	indx_arr = list(map(lambda x: name_to_index[x], arr))
	return np.array( indx_arr, dtype=np.int32 )


def load_symbol2vec(checkpoint, vocab_size, embedding_size, num_samples, learning_rate, valid_examples, SAVER_PATH):
	"""
		Restore a model to continue working on
	"""
	#valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

	x = tf.placeholder(tf.int32, shape=[None,], name="x_pivot_words")
	y = tf.placeholder(tf.int32, shape=[None,], name="y_target_words")
	embedding = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0), name="symbol_embedding")
	nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=tf.sqrt(1/embedding_size)), name="nce_weights")
	nce_biases = tf.Variable(tf.zeros([vocabulary_size], name="nce_biases"))
	## need to lookup embedding for pivot word each iteration, pass in x
	pivot = tf.nn.embedding_lookup(embedding, x, name="word_embedding_lookup")
	# set a fixed shape size for y
	train_labels = tf.reshape(y , [tf.shape(y)[0], 1])
	loss = tf.reduce_mean(tf.nn.nce_loss(	weights		= nce_weights, 
											biases		= nce_biases,
											labels		= train_labels,
											inputs		= pivot,
											num_sampled = num_neg_samples,
											num_classes = vocabulary_size,
											num_true	= 1))
	tf.summary.scalar('loss', loss)
	"""
	optimizer = tf.contrib.layers.optimize_loss(loss, 
												tf.train.get_global_step(),
												learning_rate,
												"Adam", 
												clip_gradients=5.0,
												name="optimizer")
	"""
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

	##keep track of similarity between symbol vectors
	##use cosine sim, shortcut - use A/l2(A) then multiply different A's together to get cosine sim

	# Compute the cosine similarity between minibatch examples and all embeddings.
	#norm = tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keep_dims=True))
	#normalized_embedding = embedding / norm
	## valid dataset is a list of known symbol names use to test this
	#valid_embeddings = tf.nn.embedding_lookup( normalized_embedding, valid_dataset)
	#compute similarity
	#similarity = tf.matmul( valid_embeddings, normalized_embedding, transpose_b=True)
	saver = tf.train.Saver()

	merged = tf.summary.merge_all()

	sess = tf.Session()
	logger.info("Restoring session...")

	saver.restore(sess, config.res + "/" + SAVER_PATH + "model.session.ckpt-" + str(checkpoint))
	logger.info("Restored session!")

	path = '/root/desyl/res/' + SAVER_PATH
	summary_writer = tf.summary.FileWriter(path, sess.graph)
	proj_config = projector.ProjectorConfig()
	embed = proj_config.embeddings.add()
	embed.tensor_name = embedding.name
	embed.metadata_path = path + 'metadata.tsv'

	#return optimizer, loss, x, y, sess, summary_writer, merged, similarity, normalized_embedding, embedding
	return optimizer, loss, x, y, sess, summary_writer, merged, embedding


def build_symbol2vec(vocab_size, embedding_size, num_samples, learning_rate, valid_examples, SAVER_PATH):
	"""
		num_samples is the number of samples to use in negative sampling.
	"""

	#valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

	x = tf.placeholder(tf.int32, shape=[None,], name="x_pivot_words")
	y = tf.placeholder(tf.int32, shape=[None,], name="y_target_words")
	#y = tf.placeholder(tf.int32, shape=[None,vocab_size], name="y_target_words")
	embedding = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="symbol_embedding")
	nce_weights = tf.Variable(tf.truncated_normal([vocab_size, embedding_size], stddev=tf.sqrt(1/embedding_size)), name="nce_weights")
	nce_biases = tf.Variable(tf.zeros([vocab_size], name="nce_biases"))

	## need to lookup embedding for pivot word each iteration, pass in x
	pivot = tf.nn.embedding_lookup(embedding, x, name="word_embedding_lookup")

	# set a fixed shape size for y
	train_labels = tf.reshape(y , [tf.shape(y)[0], 1])
	#train_labels = tf.reshape(y , tf.shape(y) )
	#train_labels = y


	loss = tf.reduce_mean(tf.nn.nce_loss(	weights		= nce_weights, 
											biases		= nce_biases,
											labels		= train_labels,
											inputs		= pivot,
											num_sampled = num_samples,
											num_classes = vocab_size,
											num_true	= 1))

	tf.summary.scalar('loss', loss)

	optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
	#Adam is slow as hell
	"""
	optimizer = tf.contrib.layers.optimize_loss(loss, 
												tf.train.get_global_step(),
												learning_rate,
												"Adam", 
												clip_gradients=5.0,
												name="optimizer")
	"""


	##keep track of similarity between symbol vectors
	##use cosine sim, shortcut - use A/l2(A) then multiply different A's together to get cosine sim

	# Compute the cosine similarity between minibatch examples and all embeddings.
	#norm = tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keep_dims=True))
	#normalized_embedding = embedding / norm

	## valid dataset is a list of known symbol names use to test this
	#valid_embeddings = tf.nn.embedding_lookup( normalized_embedding, valid_dataset)

	#compute similarity
	#similarity = tf.matmul( valid_embeddings, normalized_embedding, transpose_b=True)

	merged = tf.summary.merge_all()
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	path = '/root/desyl/res/' + SAVER_PATH
	summary_writer = tf.summary.FileWriter(path, sess.graph)
	proj_config = projector.ProjectorConfig()
	embed = proj_config.embeddings.add()
	embed.tensor_name = embedding.name
	embed.metadata_path = path + 'metadata.tsv'

	return optimizer, loss, x, y, sess, summary_writer, merged

def corpus_to_pivot_targets(name_to_index, corpus):
	pivots, targets = [], []

	for pivot, sentences in tqdm.tqdm(corpus.items()):
		for sentence in sentences:
			if pivot in sentence:
				sentence.remove( pivot )

			target_inds = list(map(lambda x: name_to_index[x], sentence))
			target_one_hot = np.zeros( (len(name_to_index), ), dtype=np.int32 )
			for i in target_inds:
				target_one_hot[i] = 1

			pivots.append( name_to_index[ pivot ] )
			targets.append( target_one_hot )

	return pivots, targets



def corpus_to_single_pivot_targets(name_to_index, corpus):
	pivots, targets = [], []

	for pivot, sentences in tqdm.tqdm(corpus.items()):
		for sentence in sentences:
			for symbol in sentence:
				if symbol == pivot:
					continue
				pivots.append( name_to_index[ pivot ] )
				targets.append( name_to_index[ symbol ] )

	return pivots, targets


def nearest_symbol_embeddings_vec(normalized_embedding, embedding, index_to_name, vec, top_k=10):
	"""
		Find teh nearest top_k symbols in the embedding to the vector vec
		normalized_embedding: np obj - tensorflow that has been eval()'ed.
		Cannot create a tensorflow object greater than 2GB - howver can with np obj
	"""
	vec =np.reshape(vec,(-1,1))

	#compute A/|A|
	vec_norm = tf.sqrt(tf.reduce_sum(tf.square( vec ) ))
	mod_vec = vec / vec_norm

	#compute similarity

	#compute B/|B| for all B in the corpus
	#half_cosine_sim_all_symbols = tf.div( embedding, normalized_embedding )

	#compute all angle to vec at the same time
	#this is greater than 2GB ram for tensor
	sim = normalized_embedding @ mod_vec.eval()

	#sim_real = sim.eval()
	sim_real = sim

	#reshape to array
	sim_real = np.reshape( sim_real, (-1, ) )

	nearest = (-sim_real).argsort()[0:top_k + 1]
	log_str = 'Nearest symbols:\n\t'

	#for k in range(top_k):
	for k in range(top_k+1):
		close_word = index_to_name[nearest[k]]
		log_str = '%s {%s:%f}  ' % (log_str, close_word, sim_real[nearest[k]])
	print(log_str)

def cosine_symbol_distance(embedding, name_to_index, symbol_a, symbol_b):
	vec_a = create_symbol_vector(embedding, [ symbol_a ], [], name_to_index)
	vec_b = create_symbol_vector(embedding, [ symbol_b ], [], name_to_index)

	a_norm, b_norm = np.linalg.norm(vec_a), np.linalg.norm(vec_b)
	return (vec_a @ vec_b) / (a_norm * b_norm)

def create_symbol_vector(embedding, positive, negative, name_to_index):
	"""
		:param positive: A list of symbols to be added
		:param negative: A list of symbols to be negated
	"""

	r, c = np.shape( embedding )

	vec = np.zeros( (c, ), dtype=np.float32 )
	for i in positive:
		vec += embedding[ name_to_index[i], : ]
	for i in negative:
		vec -= embedding[ name_to_index[i], : ]

	return vec

def nearest_k_symbol_vectors(similarity, index_to_name, symbol_index):
	sim = similarity.eval()

	valid_word = index_to_name[symbol_index]
	top_k = 8  # number of nearest neighbors
	nearest = (-sim[i, :]).argsort()[0:top_k + 1]
	log_str = 'Nearest to %s:\n\t' % valid_word
	#for k in range(top_k):
	for k in range(top_k+1):
		close_word = index_to_name[nearest[k]]
		log_str = '%s {%s:%f}  ' % (log_str, close_word, sim[i, nearest[k]])
	print(log_str)
	return

def remove_stop_symbols(pivot_words, target_words, stop_symbols):
	assert(isinstance(stop_symbols, set))
	assert(len(pivot_words) == len(target_words))

	_pivot_words, _target_words = [], []
	for i in tqdm.tqdm(range(len(pivot_words))):
		pword = index_to_name[ pivot_words[i] ]
		tword = index_to_name[ target_words[i] ]

		if pword not in stop_symbols and tword not in stop_symbols:
			_pivot_words.append( pivot_words[i] )
			_target_words.append( target_words[i] )

	return _pivot_words, _target_words

def __OLD_remove_stop_symbols(pivot_words, target_words, stop_symbols):
	"""
		The method of calling del list[index] icompletes in O(n - index). O(1/2n**2) max
		Far faster to create new filtered lists :(
	"""
	assert(isinstance(stop_symbols, set))
	assert(len(pivot_words) == len(target_words))
	indexes_to_remove = []
	for i in tqdm.tqdm(range(len(pivot_words))):
		pword = index_to_name[ pivot_words[i] ]
		tword = index_to_name[ target_words[i] ]

		if pword in stop_symbols or tword in stop_symbols:
			indexes_to_remove.append(i)

	print("Removing", len(indexes_to_remove), "IOs")

	sys.exit()

	for i in tqdm.tqdm(indexes_to_remove):
		del pivot_words[i]
		del target_words[i]

	return pivot_words, target_words





if __name__ == '__main__':
	config = Config()
	config.logger.setLevel(logging.INFO)
	logger = config.logger
	#db = Database(config)

	SUBSAMPLE_CORPUS = False
	TEST_MODEL = True
	LOAD_PREVIOUS_SESS = True
	#CHECKPOINT_NUM = 18032336
	CHECKPOINT_NUM = "FINAL"
	CACHE_TARGET_PIVOT = False
	FILTER_STOP_SYMBOLS = False
	SAMPLE_EPOCH_DATA = True
	ALL_EPOCH_DATA = not SAMPLE_EPOCH_DATA



	if SUBSAMPLE_CORPUS:
		import IPython
		corpus = classes.utils.load_py_obj(config, 'corpus') 
		corpus, name_to_index, index_to_name = subsample_corpus(corpus)
		corpus, name_to_index, index_to_name = remove_symbols_with_no_valid_contexts(corpus, name_to_index)

		classes.utils.save_py_obj(config, corpus, 'subsampled_corpus')
		classes.utils.save_py_obj(config, name_to_index, 'subsampled_name_to_index')
		classes.utils.save_py_obj(config, index_to_name, 'subsampled_index_to_name')
		IPython.embed()
		sys.exit()

	#### Config
	name_to_index	= classes.utils.load_py_obj(config, 'subsampled_name_to_index')
	index_to_name	= classes.utils.load_py_obj(config, 'subsampled_index_to_name')
	corpus			= classes.utils.load_py_obj(config, 'subsampled_corpus')

	vocabulary_size = len(name_to_index)
	embedding_size = 1024
	## lower than sqrt(len(name_to_index))
	#batch_size = 1024
	batch_size = 64
	skip_window = 2
	#num_neg_samples = batch_size - 1
	#original paper recommends 2-5 for large dataset and 5-20 for small, even better with 15?
	num_neg_samples = batch_size // 2
	#num_neg_samples = 15
	#learning_rate of 1.0e-5 is far too slow with lareg batch size, does nothing,
	#1.0e-4 is still too slow but ok 4700 err -> 4500 after 550/7500 batches
	#1.0e-1 is too fast, loss fluctuates after an hour and doesn't go down
	#learning_rate = 1.0e-2
	#learning_rate = 0.025
	learning_rate = 1.0

	num_epochs = 1000
	#step_size = 250
	step_size = 10000
	#save model every n steps
	#has to be a multiple of step_size. 20K steps
	save_step = 20 * step_size
	summary_step = step_size
	NUM_SAMPLES_PER_EPOCH = 100
	SAVER_PATH = "symbol2vec/"

	#valid_examples = build_examples(name_to_index)
	valid_examples = []

	if TEST_MODEL:
		#optimizer, loss, x, y, sess, summary_writer, merged, similarity, normalized_embedding, embedding = load_symbol2vec(CHECKPOINT_NUM, vocabulary_size, embedding_size, num_neg_samples, learning_rate, valid_examples, SAVER_PATH)
		optimizer, loss, x, y, sess, summary_writer, merged, embedding = load_symbol2vec(CHECKPOINT_NUM, vocabulary_size, embedding_size, num_neg_samples, learning_rate, valid_examples, SAVER_PATH)

		with sess.as_default():

			classes.utils.save_py_obj(config, embedding.eval(), 'symbol_embeddings')
			logger.info("Embeddings saved as numpy ndarry!")

			"""
			sim = similarity.eval()
			for i in range(len(valid_examples)):
				valid_word = index_to_name[valid_examples[i]]
				top_k = 8  # number of nearest neighbors
				nearest = (-sim[i, :]).argsort()[0:top_k + 1]
				log_str = 'Nearest to %s:\n\t' % valid_word
				#for k in range(top_k):
				for k in range(top_k+1):
					close_word = index_to_name[nearest[k]]
					log_str = '%s {%s:%f}  ' % (log_str, close_word, sim[i, nearest[k]])
				print(log_str)


			"""
			norm = tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keep_dims=True))
			normalized_embedding = embedding / norm

			np_embedding = embedding.eval()
			np_normalized_embedding = normalized_embedding.eval()

			logger.info("Calculating nearest vector to MD5_Init + MD5_Final - SHA512_Init...")
			### add symbols together 
			#vec = create_symbol_vector( embedding.eval(), ['md5_init_ctx', 'md5_finish_ctx'], ['sha512_init_ctx'], name_to_index)
			vec = create_symbol_vector( embedding.eval(), ['MD5_Init', 'MD5_Final'], ['SHA512_Init'], name_to_index)
			nearest_symbol_embeddings_vec(np_normalized_embedding, np_embedding, index_to_name, vec, 50) 

			frequencies = {}
			for k in tqdm.tqdm(corpus.keys()):
				count = functools.reduce(lambda x, y: x+y, corpus[k].values(), 0)
				frequencies[k] = count

			common = list(filter(lambda x: x[1] > 9000, frequencies.items()))
			#nearest_symbol_embeddings_vec(np_normalized_embedding, np_embedding, index_to_name, create_symbol_vector( embedding.eval(), ['bfd_hash_allocate'], [], name_to_index), 50)

			symbol_names = set(name_to_index.keys())

			#d = cosine_symbol_distance(np_embedding, name_to_index, 'dlr_add', 'gw_panic')
			#print("cos( <(dlr-add, gw_panic) ) == {}".format(d))

			import IPython
			IPython.embed()
			sys.exit()



	if CACHE_TARGET_PIVOT:
		logger.info("Loading all callgraphs...") 
		GG = classes.callgraph.mp_load_all_cgs(config, cg_dir='corpus_cgs')

		logger.error("Building new temporary style corpus")
		corpus = generate_symbol_sentences(GG, name_to_index, skip_window=1, combinations=True)
		import IPython
		IPython.embed()


		sys.exit()


		logger.info("Creating symbol sentences corpus...")
		#corpus = generate_symbol_sentences(GG, name_to_index)
		corpus = generate_corpus_sampler(GG, name_to_index, stop_symbols)
		classes.utils.save_py_obj(config, corpus, 'corpus')

		import IPython
		IPython.embed()

		sys.exit()
		#logger.info("Creating pivot words and target words...")
		#pivot_words, target_words = corpus_to_single_pivot_targets(name_to_index, corpus)

		#classes.utils.save_py_obj(config, pivot_words, 'pivot_words')
		#classes.utils.save_py_obj(config, target_words, 'target_words')

		#logger.info("done!")
	
	"""
	logger.info("Loading pivot and target words...")
	if FILTER_SYMBOLS:
		pivot_words		= classes.utils.load_py_obj(config, 'pivot_words')
		target_words	= classes.utils.load_py_obj(config, 'target_words')
		logger.info("Loaded!")

		logger.info("Removing stop symbols!")
		pivot_words, target_words = remove_stop_symbols(pivot_words, target_words, stop_symbols)
		sys.exit()
	"""

	#pivot_words	 = classes.utils.load_py_obj(config, 'pivot_words_filtered')
	#target_words	 = classes.utils.load_py_obj(config, 'target_words_filtered')
	corpus	  = classes.utils.load_py_obj(config, 'subsampled_corpus')
	logger.info("Loaded!")

	logger.info("Running Symbol2vec")

	# first, create a TensorFlow constant
	#const = tf.constant(2.0, name="const")
	# create TensorFlow variables
	#b = tf.Variable(2.0, name='b')


	logger.info("Creating tensorflow model")
	if LOAD_PREVIOUS_SESS:
		#optimizer, loss, x, y, sess, summary_writer, merged = load_symbol2vec(CHECKPOINT_NUM, vocabulary_size, embedding_size, num_neg_samples, learning_rate, valid_examples)
		optimizer, loss, x, y, sess, summary_writer, merged, embedding = load_symbol2vec(CHECKPOINT_NUM, vocabulary_size, embedding_size, num_neg_samples, learning_rate, valid_examples, SAVER_PATH)
	else:
		optimizer, loss, x, y, sess, summary_writer, merged = build_symbol2vec(vocabulary_size, embedding_size, num_neg_samples, learning_rate, valid_examples, SAVER_PATH)


	logger.info("Finished creating the model!")
	saver = tf.train.Saver(save_relative_paths=True)


	if ALL_EPOCH_DATA:
		logger.info("Generating all data points")
		X_train, Y_train = generate_epoch_all(corpus, name_to_index)
		X_train = list(map(lambda x: name_to_index[x], X_train))
		Y_train = list(map(lambda x: name_to_index[x], Y_train))

	logger.info("============STARTING TRAINING============")

	step_loss = 0.0
	for e in range(num_epochs):
		print("EPOCH:", e, "of", num_epochs)

		if SAMPLE_EPOCH_DATA:
			print("Generating epoch data")
			X_train, Y_train = generate_epoch_sample(corpus, name_to_index, NUM_SAMPLES_PER_EPOCH)
			#convert to numeric indexes
			X_train = list(map(lambda x: name_to_index[x], X_train))
			Y_train = list(map(lambda x: name_to_index[x], Y_train))

		print("Shuffling data")
		#shuffle all data keeping same indexed X's and Y's together
		X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0, shuffle=True)
		num_batches = len(X_train) // batch_size
		for i in tqdm.tqdm(range(num_batches)):
			if i != num_batches-1:
				x_batch = X_train[i*batch_size:(i+1)*batch_size]
				y_batch = Y_train[i*batch_size:(i+1)*batch_size]
			else:
				x_batch = X_train[i*batch_size:]
				y_batch = Y_train[i*batch_size:]

			_, l, summ = sess.run([optimizer, loss, merged], feed_dict = { x: x_batch, y: y_batch } )
			if i % summary_step:
				summary_writer.add_summary(summary=summ, global_step=i)

			step_loss += l
			if not i % step_size and i > 0:
				avg_loss = step_loss / float(step_size)
				logger.info("STEP: {} of {}, AVG STEP LOSS: {}".format( i, num_batches, avg_loss)) 
				step_loss = 0.0

				if not i % save_step:
					logger.info("Saving session")
					save_path = saver.save(sess, config.res + "/" + SAVER_PATH + "model.session.ckpt", i+(e*num_batches))

		save_path = saver.save(sess, config.res + "/" + SAVER_PATH + "model.session.ckpt-FINAL")
		#shuffle data
		#logger.info("Shuffling data.")
		#X_train, X_test, Y_train, Y_test = train_test_split(pivot_words, target_words, test_size=0.0, shuffle=True)
	logger.info("Finished all epochs!")
	import IPython
	IPython.embed()
