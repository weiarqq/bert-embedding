import tensorflow as tf
import modeling
import tokenization
from utils import word_ids
import numpy as np

class Config():
    is_training = False
    max_seq_length = 30
    batch_size = 1
    bert_config = modeling.BertConfig.from_json_file('chinese_L-12_H-768_A-12/bert_config.json')
    init_checkpoint = 'chinese_L-12_H-768_A-12/bert_model.ckpt'
    vocab = 'chinese_L-12_H-768_A-12/vocab.txt'
    layer_index = [-2]


graph = tf.Graph()
with graph.as_default():
    input_ids = tf.placeholder(dtype=tf.int32, shape=[Config.batch_size, Config.max_seq_length], name='input_ids')
    input_mask = tf.placeholder(dtype=tf.int32, shape=[Config.batch_size, Config.max_seq_length], name='input_mask')
    segment_ids = tf.placeholder(dtype=tf.int32, shape=[Config.batch_size, Config.max_seq_length], name='segment_ids')

    model = modeling.BertModel(
                   Config.bert_config,
                   Config.is_training,
                   input_ids=input_ids,
                   input_mask=input_mask,
                   token_type_ids=segment_ids,
                   use_one_hot_embeddings=False,
                   scope=None)

    #embedding = model.get_sequence_output()
    tvars = tf.trainable_variables()
    (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, Config.init_checkpoint)

    tf.train.init_from_checkpoint(Config.init_checkpoint, assignment_map)

    with tf.variable_scope('pooling'):

        if len(Config.layer_index) == 1:
            encode_layer = model.all_encoder_layers[Config.layer_index[0]]
        else:
            encode_layer = [model.all_encoder_layers[index] for index in Config.layer_index]
            encode_layer = tf.concat(encode_layer, axis=-1)

    input_mask_f = tf.cast(input_mask, dtype=tf.float32)
    input_mask_expand = tf.expand_dims(input_mask_f, axis=-1)
    mul_mask = tf.multiply(encode_layer, input_mask_expand)
    pooled = tf.reduce_sum(mul_mask, axis=1)/ (tf.reduce_sum(input_mask_f, axis=1, keep_dims=True)+1e-10)
    # mul_mask = lambda x, m: x * m
    # mask_readuce_mean = lambda x, m: tf.reduce_sum(mul_mask(x, m), axis=1) / (tf.reduce_sum(m, axis=1, keep_dims=True)+ 1e-10)
    # pooled = mask_readuce_mean(encode_layer, input_mask)
    pooled = tf.identity(pooled, name='final_encodes')
    out_put = [pooled]
    init = tf.global_variables_initializer()

    config = tf.ConfigProto(allow_soft_placement=True)
with tf.Session(graph=graph) as sess:
    sess.run(init)
    sequence = str(input())
    if isinstance(sequence, str):
        sequence = [sequence]
    tokenizer = tokenization.FullTokenizer(vocab_file=Config.vocab)
    input_ids_list, input_mask_list, segment_ids_list = word_ids(sequence, tokenizer, Config.max_seq_length)
    input_ids_seqs = np.reshape(input_ids_list, newshape=[-1, Config.batch_size, Config.max_seq_length])
    input_mask_seqs = np.reshape(input_mask_list, newshape=[-1, Config.batch_size, Config.max_seq_length])
    segment_ids_seqs = np.reshape(segment_ids_list, newshape=[-1, Config.batch_size, Config.max_seq_length])
    for i in range(len(input_ids_seqs)):
        word_embeddings = sess.run(out_put, feed_dict={input_ids: input_ids_seqs[i],
                                                           input_mask: input_mask_seqs[i],
                                                           segment_ids: segment_ids_seqs[i]})
        print(word_embeddings.shape)
        print(word_embeddings)


