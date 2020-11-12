from consumer.consumer import Consumer
from mtcnn.mtcnn import MTCNN
from facenet.facenet import Facenet
import numpy as np
import cv2
import glob
import os
import math
import argparse
import sys

facenet = Facenet('./models/facenet/facenet.pb')
mtcnn = MTCNN('./models/mtcnn/',False)

def draw_data(img,bboxes,names,distances,margin = 10):
    overlay = img.copy()
    for idx,rectangle in enumerate(bboxes):
        
        cv2.putText(img, names[idx],
                    (int(rectangle[0])-10, int(rectangle[1])-15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0,255, 0),2)
        cv2.rectangle(overlay, (int(rectangle[0])-margin, int(rectangle[1])-margin),
                        (int(rectangle[2])+margin, int(rectangle[3])+margin),
                        (0,255,125), -1)
    

    alpha = 0.5
    cv2.addWeighted(overlay, alpha, img, 1 - alpha,0, img)
    return img

def get_embeddings(image):

    # image = cv2.resize(image,(720,480))
    bboxes,_,img = mtcnn.detect_faces(image)
    faces = extract_face(image,bboxes)
    if(faces.shape[0]>0):
        embeddings = facenet.get_embeddings(faces)
        return embeddings,bboxes,img
    return [],None,img

def prepare_gallery(gallery_path):
    image_names = glob.glob(gallery_path+'/*')
    gallery_embeddings = {}
    for image_name in image_names:
        img = cv2.imread(image_name)
        base_name = os.path.basename(image_name)
        embeddings,_,_ = get_embeddings(img)
        if len(embeddings) == 0:
            print('No face detected for image',base_name)
        elif embeddings.shape[0] > 1:
            print('more than 1 one face detected for image:',base_name) 
        else:
            gallery_embeddings[base_name.split('.')[0]] = embeddings[0]
    return gallery_embeddings

def extract_face(img,bboxes,margin = 10):
    faces = []
    for x1,y1,x2,y2,_ in bboxes:
        y1 = max(0,int(y1)-margin)
        y2 = int(y2) + margin
        x1 = max(0,int(x1)-margin)
        x2 = int(x2) + margin
        face = img[y1:y2,x1:x2]
        face = cv2.resize(face,(160,160))
        faces.append(face)
    return np.array(faces)

def get_nearest_match(gallery,embeddings):
    names = []
    distances = []
    gallery_names = list(gallery.keys())
    for embedding in embeddings:
        dist = distance(np.expand_dims(embedding,axis=0),np.array(list(gallery.values())),1)
        idx = np.argmin(dist,axis = 0)
        names.append(gallery_names[idx])
        distances.append(dist[idx])
    return names,distances

def distance(embeddings1, embeddings2, distance_metric=0):
    if distance_metric==0:
        # Euclidian distance
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff),1)
    elif distance_metric==1:
        # Distance based on cosine similarity
        dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
        norm = np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
        similarity = dot / norm
        dist = np.arccos(similarity) / math.pi
    else:
        raise 'Undefined distance metric %d' % distance_metric 
        
    return dist
def main(args):
    print('preparing gallery....')
    gallery = prepare_gallery(args.gallery_path)
    print('gallery prepared')
    print('characters in gallery->',gallery.keys())

    consumer = Consumer(args.video_path)
    ret,frame = consumer.get_frame()

    # result = cv2.VideoWriter('result.mp4',cv2.VideoWriter_fourcc(*'MP4V'), 25, (frame.shape[1],frame.shape[0])) 

    while(ret):
        embeddings,bboxes,frame = get_embeddings(frame)
        if(len(embeddings) > 0):
            names,dist = get_nearest_match(gallery,embeddings)
            frame = draw_data(frame,bboxes,names,dist)
        cv2.imshow('res',frame)
        # result.write(frame)
        ret,frame = consumer.get_frame()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    consumer.release()
# result.release()
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('gallery_path', type=str, 
        help='Directory containing the gallery images')
    parser.add_argument('video_path', type=str, 
        help='Path of the input video file')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))



    

