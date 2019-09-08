"""Training a face recognizer with TensorFlow based on the FaceNet paper
FaceNet: A Unified Embedding for Face Recognition and Clustering: http://arxiv.org/abs/1503.03832
"""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import sys
import tensorflow as tf
import numpy as np
import importlib
import argparse
import support
from scipy import misc
from tensorflow.python.ops import data_flow_ops
import os

os.environ["CUDA_VISIBLE_DEVICES"]="1"

def main(args):



    network = importlib.import_module(args.model_def)

    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), subdir)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    model_dir = os.path.join(os.path.expanduser(args.models_base_dir), subdir)
    if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
        os.makedirs(model_dir)


    # Write arguments to a text file
    support.write_arguments_to_file(args, os.path.join(log_dir, 'arguments.txt'))
        
    # Store some git revision info in a text file in the log directory
    src_path,_ = os.path.split(os.path.realpath(__file__))
    support.store_revision_info(src_path, log_dir, ' '.join(sys.argv))

    np.random.seed(seed=args.seed)
    train_set = support.get_dataset(args.data_dir)
    

    with tf.Graph().as_default():
        tf.set_random_seed(args.seed)
        global_step = tf.Variable(0, trainable=False)

        # Placeholder for the learning rate
        learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')
        
        batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
        
        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
        
        image_paths_placeholder = tf.placeholder(tf.string, shape=(None,3), name='image_paths')
        labels_placeholder = tf.placeholder(tf.int64, shape=(None,3), name='labels')
        
        input_queue = data_flow_ops.FIFOQueue(capacity=100000,
                                    dtypes=[tf.string, tf.int64],
                                    shapes=[(3,), (3,)],
                                    shared_name=None, name=None)
        enqueue_op = input_queue.enqueue_many([image_paths_placeholder, labels_placeholder])

        # TODO modality begin

        train_set_modality = support.get_dataset(args.modality_data_dir)

        # TODO modality end

        nrof_preprocess_threads = 4
        images_and_labels = []
        for _ in range(nrof_preprocess_threads):
            filenames, label = input_queue.dequeue()
            images = []
            for filename in tf.unstack(filenames):
                file_contents = tf.read_file(filename)
                #TODO FIFOQueue error 20180629
                image = tf.image.decode_image(file_contents, channels=3)
                
                if args.random_crop:
                    image = tf.random_crop(image, [args.image_size, args.image_size, 3])
                else:
                    image = tf.image.resize_image_with_crop_or_pad(image, args.image_size, args.image_size)
                if args.random_flip:
                    image = tf.image.random_flip_left_right(image)
    
                #pylint: disable=no-member
                image.set_shape((args.image_size, args.image_size, 3))
                images.append(tf.image.per_image_standardization(image))
            images_and_labels.append([images, label])
    
        image_batch, labels_batch = tf.train.batch_join(
            images_and_labels, batch_size=batch_size_placeholder, 
            shapes=[(args.image_size, args.image_size, 3), ()], enqueue_many=True,
            capacity=4 * nrof_preprocess_threads * args.batch_size,
            allow_smaller_final_batch=True)
        image_batch = tf.identity(image_batch, 'image_batch')
        image_batch = tf.identity(image_batch, 'input')
        labels_batch = tf.identity(labels_batch, 'label_batch')

        # Build the inference graph
        prelogits, prelogits_modality, orthogonality_loss, orthConvNum = network.inference(image_batch, args.keep_probability,
            phase_train=phase_train_placeholder, bottleneck_layer_size=args.embedding_size,
            weight_decay=args.weight_decay)
        
        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
        # Split embeddings into anchor, positive and negative and calculate triplet loss
        anchor, positive, negative = tf.unstack(tf.reshape(embeddings, [-1,3,args.embedding_size]), 3, 1)
        triplet_loss = support.triplet_loss(anchor, positive, negative, args.alpha)

        learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step,
            args.learning_rate_decay_epochs*args.epoch_size, args.learning_rate_decay_factor, staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)


        # Create a saver
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=10)

        # TODO save best acc
        # Create a saver
        best_saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=10)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        # Start running operations on the Graph.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))        

        # Initialize variables
        sess.run(tf.global_variables_initializer(), feed_dict={phase_train_placeholder:True})



        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord, sess=sess)
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        with sess.as_default():

            if args.pretrained_model:
                print('Restoring pretrained model: %s' % args.pretrained_model)
                support.load_model(args.pretrained_model)

            img_list = []
            image_path='./VIS_sample.png'
            img = misc.imread(os.path.expanduser(image_path), mode='RGB')
            aligned = misc.imresize(img, (args.image_size, args.image_size), interp='bilinear')
            prewhitened = support.prewhiten(aligned)
            img_list.append(prewhitened)
            images = np.stack(img_list)
            feed_dict = { images_placeholder: images, phase_train_placeholder:False }
            feas = sess.run(embeddings, feed_dict=feed_dict)
            print(image_path)
            print(feas)

    return model_dir



def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--logs_base_dir', type=str,
                        help='Directory where to write event logs.',
                        default='./logs/DSVNs_result')
    parser.add_argument('--models_base_dir', type=str,
                        help='Directory where to write trained models and checkpoints.',
                        default='./models/DSVNs_model_saver')
    parser.add_argument('--best_models_base_dir', type=str,
                        help='Directory where to write trained models and checkpoints.',
                        default='./models/DSVNs_model_saver_best_models')
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=0.8)
    parser.add_argument('--pretrained_model', type=str,
                        help='Load a pretrained model before training starts.',
                        default='./')
    parser.add_argument('--data_dir', type=str,
        help='Path to the data directory containing aligned face patches.',
        default='./')
    parser.add_argument('--modality_data_dir', type=str,
        help='Path to the data directory containing aligned face patches.',
        default='./')
    parser.add_argument('--model_def', type=str,
                        help='Model definition. Points to a module models/containing the definition of the inference graph.',
                        default='DSVNs_Architecture')
    parser.add_argument('--max_nrof_epochs', type=int,
                        help='Number of epochs to run.', default=3000)
    parser.add_argument('--train_epoch_distribution', type=int,
                        help='one epoch for modality, and k epoch for identity', default=4)
    parser.add_argument('--batch_size', type=int,
                        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--people_per_batch', type=int,
                        help='Number of people per batch.', default=60)
    parser.add_argument('--images_per_person', type=int,
                        help='Number of images per person.', default=9)
    parser.add_argument('--people_per_batch_modality', type=int,
                        help='Number of people per batch.', default=2)
    parser.add_argument('--images_per_person_modality', type=int,
                        help='Number of images per person.', default=45)
    parser.add_argument('--epoch_size', type=int,
                        help='Number of batches per epoch.', default=300)  # ori 1000
    parser.add_argument('--alpha', type=float,
                        help='Positive to negative triplet distance margin.', default=0.2)
    parser.add_argument('--embedding_size', type=int,
                        help='Dimensionality of the embedding.', default=128)
    parser.add_argument('--embedding_size_modality', type=int,
                        help='Dimensionality of the embedding.', default=128)
    parser.add_argument('--random_crop',
                        help='Performs random cropping of training images. If false, the center image_size pixels from the training images are used. ' +
                             'If the size of the images in the data directory is equal to image_size no cropping is performed',
                        action='store_true')
    parser.add_argument('--random_flip',
                        help='Performs random horizontal flipping of training images.', action='store_true')
    parser.add_argument('--keep_probability', type=float,
                        help='Keep probability of dropout for the fully connected layer(s).', default=0.8)
    parser.add_argument('--weight_decay', type=float,
                        help='L2 weight regularization.', default=5e-4)
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'],
                        help='The optimization algorithm to use', default='ADAGRAD')
    parser.add_argument('--learning_rate', type=float,
                        help='Initial learning rate. If set to a negative value a learning rate ' +
                             'schedule can be specified in the file "learning_rate_schedule.txt"', default=0.001)
    parser.add_argument('--learning_rate_modality', type=float,
                        help='Initial learning rate. If set to a negative value a learning rate ' +
                             'schedule can be specified in the file "learning_rate_schedule.txt"', default=0.0001)
    parser.add_argument('--learning_rate_decay_epochs', type=int,
                        help='Number of epochs between learning rate decay.', default=2)
    parser.add_argument('--learning_rate_decay_factor', type=float,
                        help='Learning rate decay factor.', default=0.9)
    parser.add_argument('--moving_average_decay', type=float,
                        help='Exponential decay for tracking of training parameters.', default=0.9999)
    parser.add_argument('--seed', type=int,
                        help='Random seed.', default=666)
    parser.add_argument('--learning_rate_schedule_file', type=str,
                        help='File containing the learning rate schedule that is used when learning_rate is set to to -1.',
                        default='./data/learning_rate_schedule.txt')

    return parser.parse_args(argv)
  

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
