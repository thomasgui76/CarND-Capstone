from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import os.path

TRAFFIC_LIGHT_COLORS = ["GREEN", "RED", "YELLOW"]
CONFIDENT_THRESHOLD = 0.5

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        MODEL_PATH = os.path.abspath(rospy.get_param('model_name'))
        # rospy.loginfo("MODEL_PATH: {}".format(MODEL_PATH))
        self.COLOR_ARRAY = [(0, 255, 0), (255, 0, 0), (255, 255, 0)]

        self.bridge = CvBridge()
        self.image_pub = rospy.Publisher("processed_image", Image, queue_size=1)

        # Load tensorflow graph
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            loaded_graph_def = tf.GraphDef()
            with tf.gfile.GFile(MODEL_PATH, 'rb') as fid:
                serialized_graph = fid.read()
                loaded_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(loaded_graph_def, name='')

            # Get tensors from loaded graph
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

        # Get session
        config = tf.ConfigProto(log_device_placement=False)
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.5
        self.sess = tf.Session(graph=self.detection_graph, config=config)


    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        with self.detection_graph.as_default():
            # expand simention to reshape input image to [1, None, None, 3].
            img_expanded = np.expand_dims(image, axis=0)
            feed_dict = {
                self.image_tensor: img_expanded
            }
            boxes, scores, classes, num_detections = self.sess.run(
                [self.detection_boxes, self.detection_scores,
                 self.detection_classes, self.num_detections], feed_dict=feed_dict)
            outimage = image
            try:
                self.image_pub.publish(self.bridge.cv2_to_imgmsg(outimage, "rgb8"))
            except CvBridgeError as e:
                rospy.logerror(e)

            # Get the most likely color with the highest score
            color = None
            highest_score = scores[0][0]
            # rospy.logdebug("num_detections: {}\n".format(num_detections))
            # rospy.logdebug("scores: {}\n".format(scores))
            # rospy.logdebug("classes: {}\n".format(classes))
            most_likely_class = classes[0][0]
            if num_detections > 0:
                if highest_score >= CONFIDENT_THRESHOLD:
                    color = most_likely_class
                else:
                    log_msg = '# detections is {}, but confident {} is not enough'.format(
                        num_detections, highest_score)
                    rospy.loginfo(log_msg)

            if color == 1:
                rospy.loginfo('Infered traffic light is GREEN')
                return TrafficLight.GREEN
            elif color == 2:
                rospy.loginfo('Infered traffic light is RED')
                return TrafficLight.RED
            elif color == 3:
                rospy.loginfo('Infered traffic light is YELLOW')
                return TrafficLight.YELLOW
            else:
                rospy.loginfo('UNKNOWN traffic light')
                return TrafficLight.UNKNOWN
