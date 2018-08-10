import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import pandas as pd

flags = tf.app.flags
flags.DEFINE_string('MODEL_NAME','inference_graph', '')

CWD_PATH = '/home/tector/gtsdb'
flags.DEFINE_string('IMAGE_PATH', '/home/tector/data/jpg_FullIJCNN2013/', 'Path to images in jpg format')

flags.DEFINE_string('PATH_TO_CKPT', os.path.join(CWD_PATH, 'graph', 'frozen_inference_graph.pb'), '')

flags.DEFINE_string('EVAL_CSV', os.path.join(CWD_PATH, 'test_data_gtsdb.csv'), '')
FLAGS = flags.FLAGS
NUM_CLASSES = 43

threshold = 0.1

def IoU(box1, box2):
    xmin = max(box1[0], box2[0])
    ymin = max(box1[2], box2[2])
    xmax = min(box1[1], box2[1])
    ymax = min(box1[3], box2[3])
    intersection = [xmin, xmax, ymin, ymax]
   # print(intersection)
    union_area = area(box1) + area(box2) - area(intersection)
    return area(intersection)/union_area

def area(box):
    """box should be [xmin, xmax, ymin, ymax]"""
    if box[1] < box[0] or box[3] < box[2]:
        return 0;
    return (box[1]-box[0])*(box[3]-box[2])

def compute_ap(recall, precision):
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([1.], precision, [0.]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i-1] = np.maximum(mpre[i - 1], mpre[i])
        
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def compute_map(predictions, ground_truth):
    average_precisions = {}
    no_ground_truth = []
    for c in range(1,44):
        false_positives = np.zeros((0,))
        true_positives = np.zeros((0,))
        scores = np.zeros((0,))
    
        detections = predictions[predictions['id']==c]
        annotations = ground_truth[ground_truth['class']==c]
        num_annotations = annotations.shape[0]
        #print("total annotations: ", num_annotations)
        #print("total detections: ", len(detections))
        #print(annotations)
        #for i in predictions.iterrows():
    #        detections = 
        for _, detection in detections.dropna().iterrows():
            #print(detection['score'])
            scores = np.append(scores, detection['score'])
            corresponding_annotations = annotations[annotations['filename'] == detection['filename']]
            if (corresponding_annotations.empty):
                false_positives = np.append(false_positives, 1)
                true_positives = np.append(true_positives, 0)
                continue
            overlaps = np.zeros((0,))
            #print(annotations[['xmin', 'xmax', 'ymin', 'ymax']])
            for _, annotation in corresponding_annotations.iterrows():
                overlaps = np.append(overlaps, IoU(annotation[['xmin', 'xmax', 'ymin', 'ymax']], detection[['xmin', 'xmax', 'ymin', 'ymax']]))
            if max(overlaps) > 0.5:
                true_positives = np.append(true_positives, 1)
                false_positives = np.append(false_positives, 0)
            else:            
                true_positives = np.append(true_positives, 0)
                false_positives = np.append(false_positives, 1)
        if num_annotations == 0:
            no_ground_truth.append(c)
            #average_precisions[c] = 0, 0
            continue
        indices = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)
        
        recall = true_positives / num_annotations
        precision = true_positives / (true_positives + false_positives)
        
        average_precision = compute_ap(recall, precision)
        average_precisions[c] = average_precision
    print("No ground truth examples:", no_ground_truth)
    print(np.mean(list(average_precisions.values())))

    for key,value in average_precisions.items():
        print(key, ": ", value)
    
def generate_results():
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(FLAGS.PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    
        sess = tf.Session(graph=detection_graph)
    
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    
    
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    results = pd.DataFrame(columns=['filename', 'score', 'xmin', 'xmax', 'ymin', 'ymax', 'id'])
    test_data = pd.read_csv(FLAGS.EVAL_CSV)
    for filename in test_data['filename'].unique():
        image = cv2.imread(os.path.join(FLAGS.IMAGE_PATH, filename))
        height, width, _ = image.shape
        image_expanded = np.expand_dims(image, axis=0)
        (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections], feed_dict={image_tensor: image_expanded})
    
        above_threshold = np.where(scores > threshold)
        predicted_scores = scores[above_threshold]
        predicted_classes = classes[above_threshold]
        predicted_boxes = boxes[above_threshold]
        if predicted_boxes.size == 0:
            results = results.append({'filename': filename, 'score': '', 'xmin': '', 'ymin': '', 'xmax': '', 'ymax':'', 'id':''}, ignore_index=True)
        for j, box in enumerate(predicted_boxes):
            xmin = int(box[1]*width)
            xmax = int(box[3]*width)
            ymin = int(box[0]*height)
            ymax = int(box[2]*height)
            results = results.append({'filename': filename, 'score': predicted_scores[j], 'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax, 'id':int(predicted_classes[j])}, ignore_index=True)
            #cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0, 255), 1)
            #cv2.putText(image, str(int(predicted_classes[j]))+ ':' + str(int(100*predicted_scores[j])), (xmin-50, ymin-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    #    cv2.imwrite('/home/tector/gtsdb/test_' + str(i)+ '.jpg', image)
    results.to_csv('/home/tector/gtsdb/eval/eval_results.csv')
    compute_map(results, test_data)

def main(_):
    generate_results()


if __name__ == '__main__':
    tf.app.run()
