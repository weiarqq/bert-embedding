import modeling
import tokenization
import tensorflow as tf
from utils import  word_ids
import numpy as np

class Config():
    max_seq_length = 30
    batch_size = 1
    bert_config = modeling.BertConfig.from_json_file('chinese_L-12_H-768_A-12/bert_config.json')
    init_checkpoint = 'chinese_L-12_H-768_A-12/bert_model.ckpt'







input_ids = tf.placeholder(shape=[1, Config.max_seq_length], dtype=tf.int32, name="input_ids")
input_mask = tf.placeholder(shape=[1, Config.max_seq_length], dtype=tf.int32, name="input_mask")
segment_ids = tf.placeholder(shape=[1, Config.max_seq_length], dtype=tf.int32, name="segment_ids")
# 创建bert 模型
model = modeling.BertModel(
    config=Config.bert_config,
    is_training=False,
    input_ids=input_ids,
    input_mask=input_mask,  # input_mask是样本中有效词句的标识
    token_type_ids=segment_ids,  # token_type是句子标记 ##
    use_one_hot_embeddings=False
)
embedding = model.get_sequence_output()  # 获取字向量

tvars = tf.trainable_variables()  #加载bert 参数
# 加载bert 模型参数
(assignment_map, initialized_variable_names) = \
    modeling.get_assignment_map_from_checkpoint(tvars,
                                                Config.init_checkpoint)
tf.train.init_from_checkpoint(Config.init_checkpoint, assignment_map)

session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())
texts = ['在 关 系 数 据 库 中 ， 对 关 系 的 最 基 本 要 求 的 满 足 第 一 范 式']
tokenizer = tokenization.FullTokenizer(vocab_file='chinese_L-12_H-768_A-12/vocab.txt', do_lower_case=True)
input_ids_list, input_mask_list, segment_ids_list = word_ids(texts, tokenizer, Config.max_seq_length)
input_ids_list = np.reshape(input_ids_list, newshape=[-1, Config.batch_size, Config.max_seq_length])
input_mask_list = np.reshape(input_mask_list, newshape=[-1, Config.batch_size, Config.max_seq_length])
segment_ids_list = np.reshape(segment_ids_list, newshape=[-1, Config.batch_size, Config.max_seq_length])
embedding_r = session.run(embedding, feed_dict={input_ids: input_ids_list[0],
                                                   input_mask: input_mask_list[0],
                                                   segment_ids: segment_ids_list[0]})
print(type(embedding_r))
print(embedding_r.shape)
#print(embedding_r[-1][-1])


# 也可以直接加载模型
# session.run(tf.global_variables_initializer())
# saver = tf.train.Saver()
# sess.run(tf.global_variables_initializer())
# saver.restore(sess, init_checkpoint)




