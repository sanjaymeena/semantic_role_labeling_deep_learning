import math
import gzip
import paddle.v2 as paddle
import paddle.v2.evaluator as evaluator
import pi_data_feeder
import itertools
import time

# init dataset
train_data_file = 'data/predicate_identifier/train.txt'
test_data_file = 'data/predicate_identifier/test.txt'
target_file = 'data/predicate_identifier/target.txt'
vocab_file = 'data/embedding/vocab.txt'
emb_file = 'data/embedding/wordVectors.txt'

train_data_reader = pi_data_feeder.train(train_data_file, vocab_file, target_file)
test_data_reader = pi_data_feeder.test(test_data_file, vocab_file, target_file)
word_dict, label_dict = pi_data_feeder.get_dict(vocab_file, target_file)
word_vector_values = pi_data_feeder.get_embedding(emb_file)


# init hyper-params
word_dict_len = len(word_dict)
label_dict_len = len(label_dict)
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


def ner_net(is_train):
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


def ner_net_train(data_reader=train_data_reader, num_passes=5):
    # define network topology
    crf_cost, crf_dec, target = ner_net(is_train=True)
    evaluator.sum(name='error', input=crf_dec)
    evaluator.chunk(
        name='ner_chunk',
        input=crf_dec,
        label=target,
        chunk_scheme='IOB',
        num_chunk_types=(label_dict_len - 1) / 2)

    # create parameters
    parameters = paddle.parameters.create(crf_cost)
    print "total word vocab in word vector: " , len(word_vector_values)
    parameters.set('emb', word_vector_values)

    # create optimizer
    optimizer = paddle.optimizer.Momentum(
        momentum=0,
        learning_rate=2e-4,
        regularization=paddle.optimizer.L2Regularization(rate=8e-4),
        gradient_clipping_threshold=25,
        model_average=paddle.optimizer.ModelAverage(
            average_window=0.5, max_average_window=10000), )

    trainer = paddle.trainer.SGD(
        cost=crf_cost,
        parameters=parameters,
        update_equation=optimizer,
        extra_layers=crf_dec)

    reader = paddle.batch(
        paddle.reader.shuffle(data_reader, buf_size=8192), batch_size=64)

    feeding = {'word': 0, 'mark': 1, 'target': 2}

    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 100 == 0:
                print "Pass %d, Batch %d, Cost %f, %s" % (
                    event.pass_id, event.batch_id, event.cost, event.metrics)
            if event.batch_id % 1000 == 0:
                result = trainer.test(reader=reader, feeding=feeding)
                print "\nTest with Pass %d, Batch %d, %s" % (
                    event.pass_id, event.batch_id, result.metrics)

        if isinstance(event, paddle.event.EndPass):
            # save parameters
            with gzip.open('params_pass_%d.tar.gz' % event.pass_id, 'w') as f:
                parameters.to_tar(f)
            result = trainer.test(reader=reader, feeding=feeding)
            print "\nTest with Pass %d, %s" % (event.pass_id, result.metrics)

    trainer.train(
        reader=reader,
        event_handler=event_handler,
        num_passes=num_passes,
        feeding=feeding)

    return parameters





def predict_predicate(item):

    test_data=[]
    test_sentences=[]
    test_data.append([item[0], item[1]])
    test_sentences.append(item[-1])  

    gold_labels_id=item[2]
    
    lab_ids = inferer.infer(input=test_data, field='id')
    flat_data = [word for word in itertools.chain.from_iterable(test_sentences)]

    #print test_data
    #print item

    labels_reverse = {}
    for (k, v) in label_dict.items():
        labels_reverse[v] = k
    
    pre_lab = [labels_reverse[lab_id] for lab_id in lab_ids]
    gold_lab=  [labels_reverse[lab_id] for lab_id in gold_labels_id]

    #print gold_lab

    output=""
    counter=0
    for word, label in zip(flat_data, pre_lab):
        
        tlevel=word + " "  + gold_lab[counter] + " " +label 
        output+=tlevel
        output+="\n" 
        counter=counter+1
    #print output
    return output   


def ner_net_infer(data_reader=test_data_reader, model_file='ner_model.tar.gz'):
    test_data = []
    test_sentences = []
    subset=[]
    for item in data_reader():
        #print item
        subset.append(item)
        test_data.append([item[0], item[1]])
        test_sentences.append(item[-1])
        # if len(test_data) == 30:
        #     break


    builder=""
    for item in subset:
        test_data=[]
        test_sentences=[]
        test_data.append([item[0], item[1]])
        test_sentences.append(item[-1])  
        output=predict_predicate(item)
        builder+=output
        builder+="\n"

    print builder
    target = open("inference_output.txt", 'w')
    target.write(builder)
    target.close() 
   

    # test_array=[]
    # test_array.append(test_data[0])    
    # lab_ids = inferer.infer(input=test_array, field='id')


    # lab_ids = paddle.infer(
    #     output_layer=predict,
    #     parameters=paddle.parameters.Parameters.from_tar(gzip.open(model_file)),
    #     input=test_data,
    #     field='id')

    # flat_data = [word for word in itertools.chain.from_iterable(test_sentences)]

    # labels_reverse = {}
    # for (k, v) in label_dict.items():
    #     labels_reverse[v] = k
    # pre_lab = [labels_reverse[lab_id] for lab_id in lab_ids]

    # for word, label in zip(flat_data, pre_lab):
    #     print word, label


paddle.init(use_gpu=False, trainer_count=1)
model_file='models/parameters/pi/srl_pi_params_pass_390.tar.gz'
predict = ner_net(is_train=False)
parameters=paddle.parameters.Parameters.from_tar(gzip.open(model_file))
inferer = paddle.inference.Inference(
                 output_layer=predict, parameters=parameters)

if __name__ == '__main__':
    #paddle.init(use_gpu=False, trainer_count=1)
    start_time = time.time()
    ner_net_infer(
        data_reader=test_data_reader, model_file=model_file)
    print("--- %s seconds ---" % (time.time() - start_time))