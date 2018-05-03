import theano.tensor as T
import theano
import numpy as np

from spatial_transformer import SpatialTransformer
from neuralnet import metrics
from neuralnet import read_data
from neuralnet import monitor
from neuralnet import utils


def train_net(config_file):
    net = SpatialTransformer(config_file)

    X = T.tensor4(name='input_tensor')
    y = T.ivector(name='label_vector')
    X_shared = theano.shared(np.zeros(net.input_tensor_shape, 'float32'), 'input')
    y_shared = theano.shared(np.zeros(net.output_shape, 'int32'), 'output')
    lr = theano.shared(net.learning_rate, 'learning rate')

    net.set_training_status(True)
    pred = net(X)
    cost = net.build_cost(pred, y)
    updates = net.build_updates(cost, net.trainable, **{'learning_rate': lr})
    error_rate = metrics.mean_classification_error(pred, y) * 100.
    train = net.compile([], [cost, error_rate], updates=updates, givens={X: X_shared, y: y_shared})

    net.set_training_status(False)
    pred = net(X)
    trans = net.inference_trans_net(X)
    error_rate = metrics.mean_classification_error(pred, y) * 100.
    eval = net.compile([], [error_rate, trans], givens={X: X_shared, y: y_shared})

    data = read_data.load_data(net.config['data']['path'])
    dm = utils.DataManager(config_file, (X_shared, y_shared))
    dm.training_set = (data['X_train'], data['y_train'])
    dm.num_train_data = data['num_examples_train']
    dm.testing_set = (data['X_valid'], data['y_valid'])
    dm.num_test_data = data['num_examples_valid']
    mon = monitor.Monitor(config_file)
    epoch = 0
    print('Start training...')
    while epoch < net.n_epochs:
        if (epoch+1) % 20 == 0:
            new_lr = lr.get_value() * 0.7
            print('New LR: %f' % new_lr)
            lr.set_value(new_lr)

        batches = dm.get_batches(epoch, net.n_epochs, True, False)
        for X_, y_ in batches:
            dm.update_input((X_, y_))
            _cost, _error = train()
            mon.plot('training cost', _cost)
            mon.plot('training classification error', _error)
            mon.tick()

        if epoch % net.validation_frequency == 0:
            valid_batches = dm.get_batches(training=False)
            for X_, y_ in valid_batches:
                dm.update_input((X_, y_))
                _error, _trans = eval()
                mon.plot('validation classification error', _error)
            mon.save_image('transformed image', _trans[:10])
            mon.flush()
        epoch += 1
    print('Training ended.')


if __name__ == '__main__':
    train_net('spatial_transformer.config')
