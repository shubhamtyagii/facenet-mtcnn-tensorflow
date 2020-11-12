import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from mtcnn.utils.utils import get_model_filenames,detect_face, filter_bboxes
import numpy as np
import os
import cv2

class MTCNN:
    def __init__(self,weight_dir,draw_bboxes=False):
        self.draw = draw_bboxes
        tf.device('/gpu:0')
        tf.Graph().as_default()
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        
        self.sess = tf.Session(config = config)
        file_paths = get_model_filenames(weight_dir)

        saver = tf.train.import_meta_graph(file_paths[0])
        saver.restore(self.sess, file_paths[1])

        self.pnet = lambda img: self.sess.run(
            ('softmax/Reshape_1:0',
                'pnet/conv4-2/BiasAdd:0'),
            feed_dict={
                'Placeholder:0': img})

        self.rnet = lambda img: self.sess.run(
            ('softmax_1/softmax:0',
                'rnet/conv5-2/rnet/conv5-2:0'),
            feed_dict={
                'Placeholder_1:0': img})

        self.onet = lambda img: self.sess.run(
            ('softmax_2/softmax:0',
                'onet/conv6-2/onet/conv6-2:0',
                'onet/conv6-3/onet/conv6-3:0'),
            feed_dict={
                'Placeholder_2:0': img})

    def detect_faces(self,img):
        rectangles, points = detect_face(img,40,self.pnet,self.rnet,self.onet,[0.85,0.85,0.85],0.7)
        rectangles = filter_bboxes(rectangles)
        return rectangles,points,img
    
    

