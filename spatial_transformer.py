import numpy as np

from neuralnet import layers, model


class SpatialTransformer(model.Model):
    def __init__(self, config_file, **kwargs):
        super(SpatialTransformer, self).__init__(config_file)
        self.input_shape = self.config['model']['input_shape']
        self.output_shape = self.config['model']['output_shape']

        self.input_tensor_shape = [self.batch_size] + [self.input_shape[2]] + self.input_shape[:2]

        self.loc_net = layers.Sequential()
        self.trans_net = layers.Sequential()
        self.class_net = layers.Sequential()

        self.loc_net.append(layers.PoolingLayer(self.input_tensor_shape, (2, 2), layer_name='loc_net_pool1'))
        self.loc_net.append(layers.ConvolutionalLayer(self.loc_net.output_shape, 20, 5, 'uniform', layer_name='loc_net_conv1'))
        self.loc_net.append(layers.PoolingLayer(self.loc_net.output_shape, (2, 2), layer_name='loc_net_pool2'))
        self.loc_net.append(layers.ConvolutionalLayer(self.loc_net.output_shape, 20, 5, 'uniform', layer_name='loc_net_conv2'))
        self.loc_net.append(layers.FullyConnectedLayer(self.loc_net.output_shape, 50, 'uniform', 'relu', layer_name='loc_net_fc1'))
        self.loc_net.append(layers.FullyConnectedLayer(self.loc_net.output_shape, 6, activation='linear', layer_name='loc_net_fc2'))

        b = np.zeros((2, 3), 'float32')
        b[0, 0] = 1
        b[1, 1] = 1
        b = b.flatten()
        self.loc_net[-1].W.set_value(self.loc_net[-1].W.get_value() * 0.)
        self.loc_net[-1].b.set_value(b)

        self.trans_net.append(layers.TransformerLayer((self.input_tensor_shape, self.loc_net.output_shape), 3, layer_name='trans_net'))

        self.class_net.append(layers.ConvolutionalLayer(self.trans_net.output_shape, 32, 3, layer_name='class_net_conv1'))
        self.class_net.append(layers.PoolingLayer(self.class_net.output_shape, (2, 2), layer_name='class_net_pool1'))
        self.class_net.append(layers.ConvolutionalLayer(self.class_net.output_shape, 32, 3, layer_name='class_net_conv2'))
        self.class_net.append(layers.PoolingLayer(self.class_net.output_shape, (2, 2), layer_name='class_net_pool2'))
        self.class_net.append(layers.FullyConnectedLayer(self.class_net.output_shape, 256, layer_name='class_net_fc1'))
        self.class_net.append(layers.FullyConnectedLayer(self.class_net.output_shape, self.output_shape, activation='softmax',
                                                         layer_name='class_net_output'))

        self.model = self.loc_net + self.trans_net + self.class_net
        super(SpatialTransformer, self).get_all_params()
        super(SpatialTransformer, self).get_trainable()
        super(SpatialTransformer, self).get_regularizable()
        super(SpatialTransformer, self).show()

    def inference_trans_net(self, input):
        theta = self.loc_net(input)
        output = self.trans_net((input, theta))
        return output

    def inference(self, input):
        theta = self.loc_net(input)
        output = self.trans_net((input, theta))
        output = self.class_net(output)
        return output
