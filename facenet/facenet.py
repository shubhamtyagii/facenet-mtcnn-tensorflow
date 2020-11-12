import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
import numpy as np
from tensorflow.python.platform import gfile
import cv2
class Facenet:


    def __init__(self,model_filepath):
        tf.device('/gpu:0')
        print('Loading model...')
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        self.graph = tf.Graph().as_default()
        self.sess = tf.Session(config = config)
        self.load_model(model_filepath)
        self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        # print(tf.shape(self.images_placeholder))

    def load_model(self,model, input_map=None):
       
        with gfile.FastGFile(model,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, input_map=input_map, name='')
    
    def get_embeddings(self,images):
        images = self.preprocess(images)
        feed_dict = {self.images_placeholder:images,self.phase_train_placeholder:False}
        embeddings = self.sess.run(self.embeddings,feed_dict=feed_dict)
        return embeddings

    def preprocess(self,images):
        res = []
        for img in images:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = self.prewhiten(img)
            res.append(img)
        return np.array(res)

    def prewhiten(self,x):
        mean = np.mean(x)
        std = np.std(x)
        std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
        y = np.multiply(np.subtract(x, mean), 1/std_adj)
        return y  