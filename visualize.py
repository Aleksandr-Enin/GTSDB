import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import pandas as pd

threshold = 0.4

flags = tf.app.flags
flags.DEFINE_string('image_name', '', 'Path to the image')
flags.DEFINE_string('graph_path', '', 'Path to the graph')
flags.DEFINE_string('output_image', 'visualized.jpg', 'Output path')
FLAGS = flags.FLAGS

def main(_):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(FLAGS.graph_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.Session(graph=detection_graph)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
  
    
    image = cv2.imread(FLAGS.image_name)
    height, width, _ = image.shape
    image_expanded = np.expand_dims(image, axis=0)
     
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections], feed_dict={image_tensor: image_expanded})
    above_threshold = np.where(scores > threshold)
    predicted_scores = scores[above_threshold]
    predicted_classes = classes[above_threshold]
    predicted_boxes = boxes[above_threshold]
    for j, box in enumerate(predicted_boxes):
        xmin = int(box[1]*width)
        xmax = int(box[3]*width)
        ymin = int(box[0]*height)
        ymax = int(box[2]*height)

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0, 255), 1)
        cv2.putText(image, str(int(predicted_classes[j]))+ ':' + str(int(100*predicted_scores[j])), (xmin-50, ymin-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    cv2.imwrite(FLAGS.output_image, image)


if __name__ == '__main__':
    tf.app.run()

