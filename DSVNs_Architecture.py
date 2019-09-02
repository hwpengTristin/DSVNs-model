# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Contains the definition of the Inception Resnet V1 architecture.
As described in http://arxiv.org/abs/1602.07261.
  Inception-v4, Inception-ResNet and the Impact of Residual Connections
    on Learning
  Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

#DPMISNs
weight_decay=5e-4
global orthogonality_loss
orthogonality_loss=0.0
conv_conter=0
TPDs_collection='TPDs'

# Inception-Renset-A
def block35(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    """Builds the 35x35 resnet block."""
    with tf.variable_scope(scope, 'Block35', [net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
            tower_conv = slim.conv2d(net, 32, 1, scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv2d(tower_conv1_0, 32, 3, scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_2'):
            tower_conv2_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
            tower_conv2_1 = slim.conv2d(tower_conv2_0, 32, 3, scope='Conv2d_0b_3x3')
            tower_conv2_2 = slim.conv2d(tower_conv2_1, 32, 3, scope='Conv2d_0c_3x3')
        mixed = tf.concat([tower_conv, tower_conv1_1, tower_conv2_2], 3)
        up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                         activation_fn=None, scope='Conv2d_1x1')
        net += scale * up
        if activation_fn:
            net = activation_fn(net)
    return net

# Inception-Renset-B
def block17(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    """Builds the 17x17 resnet block."""
    with tf.variable_scope(scope, 'Block17', [net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
            tower_conv = slim.conv2d(net, 128, 1, scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = slim.conv2d(net, 128, 1, scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv2d(tower_conv1_0, 128, [1, 7],
                                        scope='Conv2d_0b_1x7')
            tower_conv1_2 = slim.conv2d(tower_conv1_1, 128, [7, 1],
                                        scope='Conv2d_0c_7x1')
        mixed = tf.concat([tower_conv, tower_conv1_2], 3)
        up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                         activation_fn=None, scope='Conv2d_1x1')
        net += scale * up
        if activation_fn:
            net = activation_fn(net)
    return net


# Inception-Resnet-C
def block8(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    """Builds the 8x8 resnet block."""
    with tf.variable_scope(scope, 'Block8', [net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
            tower_conv = slim.conv2d(net, 192, 1, scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = slim.conv2d(net, 192, 1, scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv2d(tower_conv1_0, 192, [1, 3],
                                        scope='Conv2d_0b_1x3')
            tower_conv1_2 = slim.conv2d(tower_conv1_1, 192, [3, 1],
                                        scope='Conv2d_0c_3x1')
        mixed = tf.concat([tower_conv, tower_conv1_2], 3)
        up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                         activation_fn=None, scope='Conv2d_1x1')
        net += scale * up
        if activation_fn:
            net = activation_fn(net)
    return net
  
def reduction_a(net, k, l, m, n):
    with tf.variable_scope('Branch_0'):
        tower_conv = slim.conv2d(net, n, 3, stride=2, padding='VALID',
                                 scope='Conv2d_1a_3x3')
    with tf.variable_scope('Branch_1'):
        tower_conv1_0 = slim.conv2d(net, k, 1, scope='Conv2d_0a_1x1')
        tower_conv1_1 = slim.conv2d(tower_conv1_0, l, 3,
                                    scope='Conv2d_0b_3x3')
        tower_conv1_2 = slim.conv2d(tower_conv1_1, m, 3,
                                    stride=2, padding='VALID',
                                    scope='Conv2d_1a_3x3')
    with tf.variable_scope('Branch_2'):
        tower_pool = slim.max_pool2d(net, 3, stride=2, padding='VALID',
                                     scope='MaxPool_1a_3x3')
    net = tf.concat([tower_conv, tower_conv1_2, tower_pool], 3)
    return net

def reduction_b(net):
    with tf.variable_scope('Branch_0'):
        tower_conv = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
        tower_conv_1 = slim.conv2d(tower_conv, 384, 3, stride=2,
                                   padding='VALID', scope='Conv2d_1a_3x3')
    with tf.variable_scope('Branch_1'):
        tower_conv1 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
        tower_conv1_1 = slim.conv2d(tower_conv1, 256, 3, stride=2,
                                    padding='VALID', scope='Conv2d_1a_3x3')
    with tf.variable_scope('Branch_2'):
        tower_conv2 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
        tower_conv2_1 = slim.conv2d(tower_conv2, 256, 3,
                                    scope='Conv2d_0b_3x3')
        tower_conv2_2 = slim.conv2d(tower_conv2_1, 256, 3, stride=2,
                                    padding='VALID', scope='Conv2d_1a_3x3')
    with tf.variable_scope('Branch_3'):
        tower_pool = slim.max_pool2d(net, 3, stride=2, padding='VALID',
                                     scope='MaxPool_1a_3x3')
    net = tf.concat([tower_conv_1, tower_conv1_1,
                        tower_conv2_2, tower_pool], 3)
    return net

def inference(images, keep_probability, phase_train=True, 
              bottleneck_layer_size=128, weight_decay=0.0, reuse=None):
    batch_norm_params = {
        # Decay for the moving averages.
        'decay': 0.995,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
        # force in-place updates of mean and variance estimates
        'updates_collections': None,
        # Moving averages ends up in the trainable variables collection
        'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],
    }
    
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        return inception_resnet_v1(images, is_training=phase_train,
              dropout_keep_prob=keep_probability, bottleneck_layer_size=bottleneck_layer_size, reuse=reuse)


def inception_resnet_v1(inputs, is_training=True,
                        dropout_keep_prob=0.8,
                        bottleneck_layer_size=128,
                        reuse=None, 
                        scope='InceptionResnetV1'):
    """Creates the Inception Resnet V1 model.
    Args:
      inputs: a 4-D tensor of size [batch_size, height, width, 3].
      num_classes: number of predicted classes.
      is_training: whether is training or not.
      dropout_keep_prob: float, the fraction to keep before final layer.
      reuse: whether or not the network and its variables should be reused. To be
        able to reuse 'scope' must be given.
      scope: Optional variable_scope.
    Returns:
      logits: the logits outputs of the model.
      end_points: the set of end_points from the inception model.
    """
    end_points = {}
  
    with tf.variable_scope(scope, 'InceptionResnetV1', [inputs], reuse=reuse):
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                                stride=1, padding='SAME'):
      
                # 149 x 149 x 32
                net = slim.conv2d(inputs, 32, 3, stride=2, padding='VALID',
                                  scope='Conv2d_1a_3x3')
                end_points['Conv2d_1a_3x3'] = net
                # 147 x 147 x 32
                net = slim.conv2d(net, 32, 3, padding='VALID',
                                  scope='Conv2d_2a_3x3')
                end_points['Conv2d_2a_3x3'] = net
                # 147 x 147 x 64
                net = slim.conv2d(net, 64, 3, scope='Conv2d_2b_3x3')
                end_points['Conv2d_2b_3x3'] = net
                # 73 x 73 x 64
                net = slim.max_pool2d(net, 3, stride=2, padding='VALID',
                                      scope='MaxPool_3a_3x3')
                end_points['MaxPool_3a_3x3'] = net
                # 73 x 73 x 80
                net = slim.conv2d(net, 80, 1, padding='VALID',
                                  scope='Conv2d_3b_1x1')
                end_points['Conv2d_3b_1x1'] = net
                # 71 x 71 x 192
                net = slim.conv2d(net, 192, 3, padding='VALID',
                                  scope='Conv2d_4a_3x3')
                end_points['Conv2d_4a_3x3'] = net
                # 35 x 35 x 256
                net = slim.conv2d(net, 256, 3, stride=2, padding='VALID',
                                  scope='Conv2d_4b_3x3')
                end_points['Conv2d_4b_3x3'] = net
                
                # 5 x Inception-resnet-A
                net = slim.repeat(net, 5, block35, scale=0.17)
                end_points['Mixed_5a'] = net

                print('5 x Inception-resnet-A', net)

                #TODO DPMISNs TPD block 1
                num_orth_vonv = [64,128,256]

                with tf.variable_scope('new_para'):
                    with tf.variable_scope('TPD_block_1'):

                        output_ID, output_modality, orthogonality_loss=Orth_block(net, num_orth_vonv[0])


                        TPD_block_1_modality = output_modality
                print('TPD_block_1_modality', TPD_block_1_modality)


                net = net + output_ID

                # Reduction-A
                with tf.variable_scope('Mixed_6a'):
                    net = reduction_a(net, 192, 192, 256, 384)
                end_points['Mixed_6a'] = net
                
                # 10 x Inception-Resnet-B
                net = slim.repeat(net, 10, block17, scale=0.10)
                end_points['Mixed_6b'] = net

                # TODO DPMISNs TPD block 2
                print('10 x Inception-Resnet-B', net)
                with tf.variable_scope('new_para'):
                    with tf.variable_scope('TPD_block_2'):

                        output_ID, output_modality, orthogonality_loss = Orth_block(net, num_orth_vonv[1])

                        TPD_block_2_modality = output_modality
                        print('TPD_block_2_modality', TPD_block_2_modality)


                net = net + output_ID
                # Reduction-B
                with tf.variable_scope('Mixed_7a'):
                    net = reduction_b(net)
                end_points['Mixed_7a'] = net
                
                # 5 x Inception-Resnet-C
                net = slim.repeat(net, 5, block8, scale=0.20)
                end_points['Mixed_8a'] = net


                # TODO DPMISNs TPD block 3
                print('5 x Inception-Resnet-C', net)
                with tf.variable_scope('new_para'):
                    with tf.variable_scope('TPD_block_3'):

                        output_ID, output_modality, orthogonality_loss = Orth_block(net, num_orth_vonv[2])

                        TPD_block_3_modality = output_modality
                        print('TPD_block_3_modality', TPD_block_3_modality)


                net = net + output_ID

                net = block8(net, activation_fn=None)
                end_points['Mixed_8b'] = net

                with tf.variable_scope('Logits'):
                    end_points['PrePool'] = net
                    #pylint: disable=no-member
                    net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID',
                                          scope='AvgPool_1a_8x8')
                    net = slim.flatten(net)
          
                    net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                                       scope='Dropout')
          
                    end_points['PreLogitsFlatten'] = net
                
                net = slim.fully_connected(net, bottleneck_layer_size, activation_fn=None, 
                        scope='Bottleneck', reuse=False)

                with tf.variable_scope('new_para'):
                    with tf.variable_scope('modality_para'):

                        with tf.variable_scope('modality_concat'):
                            #TODO modality TPD block 1 and block 2

                            modality_list = []


                            TPD_block_1_modality = slim.avg_pool2d(TPD_block_1_modality,
                                                                   TPD_block_1_modality.get_shape()[1:3],
                                                                   padding='VALID', scope='AvgPool_1')
                            print('TPD_block_1_modality',TPD_block_1_modality)


                            TPD_block_2_modality = slim.avg_pool2d(TPD_block_2_modality,
                                                                   TPD_block_2_modality.get_shape()[1:3],
                                                                   padding='VALID', scope='AvgPool_2')
                            print('TPD_block_2_modality', TPD_block_2_modality)


                            TPD_block_3_modality = slim.avg_pool2d(TPD_block_3_modality,
                                                                   TPD_block_3_modality.get_shape()[1:3],
                                                                   padding='VALID', scope='AvgPool_3')
                            print('TPD_block_3_modality', TPD_block_3_modality)
                            modality_list.append(TPD_block_1_modality)
                            modality_list.append(TPD_block_2_modality)
                            modality_list.append(TPD_block_3_modality)
                            modality_fea = tf.concat(modality_list, axis=-1)



                        with tf.variable_scope('modality_Logits'):

                            modality_fea = slim.flatten(modality_fea)

                            modality_bottleneck_layer_size=128
                            modality_fea = slim.fully_connected(modality_fea, modality_bottleneck_layer_size, activation_fn=None,
                                                              weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                                              weights_regularizer=slim.l2_regularizer(weight_decay),
                                                              scope='Bottleneck_modality_hidden', reuse=False)
                            print('modality_fea', modality_fea)
                            modality_bottleneck_layer_size=128
                            modality_fea = slim.fully_connected(modality_fea, modality_bottleneck_layer_size, activation_fn=None,
                                                              weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                                              weights_regularizer=slim.l2_regularizer(weight_decay),
                                                              scope='Bottleneck_modality', reuse=False)
                            print('modality_fea', modality_fea)

    # return net, end_points
    return net, modality_fea, tf.reshape(orthogonality_loss,[]),conv_conter


##### DPMISNs tools

def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)

def lagrange_variable(shape=[1]):
    with tf.variable_scope("lagrange_variable_"+str(conv_conter)) as scope:
        initial = tf.constant(0.001, shape=shape)
        return tf.Variable(initial)


def conv2d(input, in_c, out_c, kernel_size, strides, with_bias=True,padding='SAME'):
    with tf.variable_scope("conv") as scope:
        W = [kernel_size, kernel_size, in_c, out_c]
        S = [1, strides, strides, 1]
        global conv_conter
        kernel = tf.get_variable(name="W"+str(conv_conter), shape=W, dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1),regularizer=slim.l2_regularizer(weight_decay))
        conv_conter+=1
        conv = tf.nn.conv2d(input, kernel, S, padding=padding)
        tf.add_to_collection(TPDs_collection,kernel)
        if with_bias:
            biasVar=bias_variable([out_c])
            tf.add_to_collection(TPDs_collection, biasVar)
            return conv + biasVar
        return conv

def conv2d_regularization(input, in_c, out_c, kernel_size, strides, with_bias=True,padding='SAME'):
    with tf.variable_scope("conv") as scope:
        W = [kernel_size, kernel_size, in_c, out_c]
        S = [1, strides, strides, 1]
        global conv_conter
        kernel = tf.get_variable(name="W"+str(conv_conter), shape=W, dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1),regularizer=slim.l2_regularizer(weight_decay))
        tf.add_to_collection(TPDs_collection, kernel)
        conv_conter+=1
        conv = tf.nn.conv2d(input, kernel, S, padding=padding)
        kernel_vector=tf.reshape(kernel,[kernel_size*kernel_size,1])
        if with_bias:
            biasVar=bias_variable([out_c])
            tf.add_to_collection(TPDs_collection, biasVar)
            return conv + biasVar,kernel_vector

        return conv,kernel_vector




def _seperation_convolution_block(input, filter_increment,kernel=3, stride=1):

    '''
        Adds a grouped convolution block
        cardinality is the variable used to decide number of groups
    '''
    global orthogonality_loss
    identy_list = []
    modality_list = []


    for c in range(filter_increment):

        # identy
        x = input[:,:,:,c:(c + 1)]
        with tf.variable_scope('identity_para'):
            identy, kernel_vector1 = conv2d_regularization(x, in_c=1, out_c=1, kernel_size=kernel, strides=stride,with_bias=False)
            identy_list.append(identy)

        with tf.variable_scope('modality_para'):
            # modality
            modality,kernel_vector2 = conv2d_regularization(x, in_c=1, out_c=1, kernel_size=kernel, strides=stride,with_bias=False)
            modality_list.append(modality)

            lagrange_var=lagrange_variable()

        # orthogonality_loss=orthogonality_loss+orth_alpha*tf.square(tf.matmul(tf.transpose(kernel_vector1, perm=[1, 0]), kernel_vector2))
        orthogonality_loss = tf.add(orthogonality_loss,tf.multiply(lagrange_var,tf.matmul(tf.transpose(kernel_vector1, perm=[1, 0]), kernel_vector2)))
        # print('tf.transpose(kernel_vector1, perm=[1, 0])',tf.transpose(kernel_vector1, perm=[1, 0]))
        # print('kernel_vector2', kernel_vector2)
    with tf.variable_scope('identity_para'):
        identy_merge = tf.concat(identy_list, axis=-1)
        identy_merge = tf.nn.relu(identy_merge)
    with tf.variable_scope('modality_para'):
        modality_merge = tf.concat(modality_list, axis=-1)
        modality_merge = tf.nn.relu(modality_merge)
    return identy_merge,modality_merge,orthogonality_loss

def Orth_block(input,num_orth_vonv):
    with tf.variable_scope('share_para'):
        in_c=input.get_shape()[-1]
        out_c=input.get_shape()[-1]
        x = conv2d(input, in_c=in_c, out_c=out_c, kernel_size=1, strides=1,padding='SAME')
        print('share_para x',x)
        x = tf.nn.relu(x)
        x = conv2d(x, in_c=in_c, out_c=out_c, kernel_size=3, strides=1, padding='SAME')
        print('share_para x', x)
        x = tf.nn.relu(x)
        x = conv2d(x, in_c=in_c, out_c=num_orth_vonv, kernel_size=1, strides=1, padding='SAME')
        print('share_para x', x)
        x = tf.nn.relu(x)
    with tf.variable_scope('specific_para'):
        output_ID, output_modality, orthogonality_loss =_seperation_convolution_block(x, num_orth_vonv, kernel=3, stride=1)

    print('output_ID', output_ID)

    with tf.variable_scope('modality_para'):
        output_modality = conv2d(output_modality, in_c=num_orth_vonv, out_c=num_orth_vonv, kernel_size=1, strides=1, padding='SAME')
        output_modality = tf.nn.relu(output_modality)
    with tf.variable_scope('identity_para'):
        output_ID = conv2d(output_ID, in_c=num_orth_vonv, out_c=in_c, kernel_size=1, strides=1, padding='SAME')
        output_ID = tf.nn.relu(output_ID)
        print('output_ID', output_ID)
    return output_ID,output_modality,orthogonality_loss

