import math
import gzip
import paddle.v2 as paddle
import paddle.v2.evaluator as evaluator
import pi_data_feeder
import itertools
import time


mark_dict_len = 2
word_dim = 50
mark_dim = 5
hidden_dim = 300

mix_hidden_lr = 1e-3
default_std = 1 / math.sqrt(hidden_dim) / 3.0
emb_para = paddle.attr.Param(
    name='emb', initial_std=math.sqrt(1. / word_dim), is_static=True)
std_0 = paddle.attr.Param(initial_std=0.)
std_default = paddle.attr.Param(initial_std=default_std)


def d_type(size):
    return paddle.data_type.integer_value_sequence(size)


def predicate_identifier_net(word_dict_len,label_dict_len,is_train=False):
    word = paddle.layer.data(name='word', type=d_type(word_dict_len))
    mark = paddle.layer.data(name='mark', type=d_type(mark_dict_len))

    word_embedding = paddle.layer.mixed(
        name='word_embedding',
        size=word_dim,
        input=paddle.layer.table_projection(input=word, param_attr=emb_para))
    mark_embedding = paddle.layer.mixed(
        name='mark_embedding',
        size=mark_dim,
        input=paddle.layer.table_projection(input=mark, param_attr=std_0))
    emb_layers = [word_embedding, mark_embedding]

    word_caps_vector = paddle.layer.concat(
        name='word_caps_vector', input=emb_layers)
    hidden_1 = paddle.layer.mixed(
        name='hidden1',
        size=hidden_dim,
        act=paddle.activation.Tanh(),
        bias_attr=std_default,
        input=[
            paddle.layer.full_matrix_projection(
                input=word_caps_vector, param_attr=std_default)
        ])

    rnn_para_attr = paddle.attr.Param(initial_std=0.0, learning_rate=0.1)
    hidden_para_attr = paddle.attr.Param(
        initial_std=default_std, learning_rate=mix_hidden_lr)

    rnn_1_1 = paddle.layer.recurrent(
        name='rnn1-1',
        input=hidden_1,
        act=paddle.activation.Relu(),
        bias_attr=std_0,
        param_attr=rnn_para_attr)
    rnn_1_2 = paddle.layer.recurrent(
        name='rnn1-2',
        input=hidden_1,
        act=paddle.activation.Relu(),
        reverse=1,
        bias_attr=std_0,
        param_attr=rnn_para_attr)

    hidden_2_1 = paddle.layer.mixed(
        name='hidden2-1',
        size=hidden_dim,
        bias_attr=std_default,
        act=paddle.activation.STanh(),
        input=[
            paddle.layer.full_matrix_projection(
                input=hidden_1, param_attr=hidden_para_attr),
            paddle.layer.full_matrix_projection(
                input=rnn_1_1, param_attr=rnn_para_attr)
        ])
    hidden_2_2 = paddle.layer.mixed(
        name='hidden2-2',
        size=hidden_dim,
        bias_attr=std_default,
        act=paddle.activation.STanh(),
        input=[
            paddle.layer.full_matrix_projection(
                input=hidden_1, param_attr=hidden_para_attr),
            paddle.layer.full_matrix_projection(
                input=rnn_1_2, param_attr=rnn_para_attr)
        ])

    rnn_2_1 = paddle.layer.recurrent(
        name='rnn2-1',
        input=hidden_2_1,
        act=paddle.activation.Relu(),
        reverse=1,
        bias_attr=std_0,
        param_attr=rnn_para_attr)
    rnn_2_2 = paddle.layer.recurrent(
        name='rnn2-2',
        input=hidden_2_2,
        act=paddle.activation.Relu(),
        bias_attr=std_0,
        param_attr=rnn_para_attr)

    hidden_3 = paddle.layer.mixed(
        name='hidden3',
        size=hidden_dim,
        bias_attr=std_default,
        act=paddle.activation.STanh(),
        input=[
            paddle.layer.full_matrix_projection(
                input=hidden_2_1, param_attr=hidden_para_attr),
            paddle.layer.full_matrix_projection(
                input=rnn_2_1,
                param_attr=rnn_para_attr), paddle.layer.full_matrix_projection(
                    input=hidden_2_2, param_attr=hidden_para_attr),
            paddle.layer.full_matrix_projection(
                input=rnn_2_2, param_attr=rnn_para_attr)
        ])

    output = paddle.layer.mixed(
        name='output',
        size=label_dict_len,
        bias_attr=False,
        input=[
            paddle.layer.full_matrix_projection(
                input=hidden_3, param_attr=std_default)
        ])

    if is_train:
        target = paddle.layer.data(name='target', type=d_type(label_dict_len))

        crf_cost = paddle.layer.crf(
            size=label_dict_len,
            input=output,
            label=target,
            param_attr=paddle.attr.Param(
                name='crfw',
                initial_std=default_std,
                learning_rate=mix_hidden_lr))

        crf_dec = paddle.layer.crf_decoding(
            size=label_dict_len,
            input=output,
            label=target,
            param_attr=paddle.attr.Param(name='crfw'))

        return crf_cost, crf_dec, target
    else:
        predict = paddle.layer.crf_decoding(
            size=label_dict_len,
            input=output,
            param_attr=paddle.attr.Param(name='crfw'))

        return predict



if __name__ == '__main__':
   
    start_time = time.time()
    print("--- %s seconds ---" % (time.time() - start_time))