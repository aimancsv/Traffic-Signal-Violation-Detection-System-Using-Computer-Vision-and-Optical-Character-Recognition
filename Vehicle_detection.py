# Import necessary libraries and modules
from keras.layers import Conv2D, Input, BatchNormalization, LeakyReLU, ZeroPadding2D, UpSampling2D
from keras.layers import add, concatenate
from keras.models import Model
import numpy as np
import struct
import cv2

# Import custom modules and functions
from math_util import interval_overlap, BoundBox, intersection
from number_plate_detection import detect_number_plate

# Class to read weights from a file
class WeightReader:
    def __init__(self, weight_file):
        # Open the weight file in binary mode
        with open(weight_file, 'rb') as w_f:
            # Read the version information
            major, = struct.unpack('i', w_f.read(4))
            minor, = struct.unpack('i', w_f.read(4))
            revision, = struct.unpack('i', w_f.read(4))

            # Read the number of layers
            if (major * 10 + minor) >= 2 and major < 1000 and minor < 1000:
                w_f.read(8)
            else:
                w_f.read(4)

            # Read the rest of the weights as binary data.
            binary = w_f.read()

        # Initialize offset and store all weights in a numpy array
        self.offset = 0
        self.all_weights = np.frombuffer(binary, dtype='float32')

    def read_bytes(self, size):
        # Read bytes from weights array
        self.offset = self.offset + size
        return self.all_weights[self.offset - size:self.offset]

    def load_weights(self, model):
        # Load weights into the model layer by layer
        for i in range(106):
            try:
                conv_layer = model.get_layer('conv_' + str(i))

                # For all layers except the last three, we have batch normalization after the convolution.
                if i not in [81, 93, 105]:
                    norm_layer = model.get_layer('bnorm_' + str(i))

                    size = np.prod(norm_layer.get_weights()[0].shape)

                    beta = self.read_bytes(size)  # bias
                    gamma = self.read_bytes(size)  # scale
                    mean = self.read_bytes(size)  # mean
                    var = self.read_bytes(size)  # variance

                    weights = norm_layer.set_weights([gamma, beta, mean, var])

                # All convolutional layers have bias, and the last three also have kernel weights.
                if len(conv_layer.get_weights()) > 1:
                    bias = self.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
                    kernel = self.read_bytes(np.prod(conv_layer.get_weights()[0].shape))

                    # We reshape the kernel weights to match the shape of the weights in the layer.
                    kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
                    kernel = kernel.transpose([2, 3, 1, 0])
                    conv_layer.set_weights([kernel, bias])
                else:
                    kernel = self.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
                    kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
                    kernel = kernel.transpose([2, 3, 1, 0])
                    conv_layer.set_weights([kernel])
            except ValueError:
                pass

    def reset(self):
        # Reset the offset to 0
        self.offset = 0

# Utility function for constructing a convolutional block with optional shortcut connections and various types of layers.
def _conv_block(inp, convs, skip=True):
    x = inp
    count = 0

    # Iterate over the layers in the block. If the block should have a shortcut, save the output of the penultimate layer.
    for conv in convs:
        if count == (len(convs) - 2) and skip:
            skip_connection = x
        count += 1

        # Zero padding is added for all layers that have a stride greater than 1.
        if conv['stride'] > 1: x = ZeroPadding2D(((1, 0), (1, 0)))(x)  # peculiar padding as darknet prefer left and top
        x = Conv2D(conv['filter'],
                   conv['kernel'],
                   strides=conv['stride'],
                   padding='valid' if conv['stride'] > 1 else 'same',  # peculiar padding as darknet prefer left and top
                   name='conv_' + str(conv['layer_idx']),
                   use_bias=False if conv['bnorm'] else True)(x)
        if conv['bnorm']: x = BatchNormalization(epsilon=0.001, name='bnorm_' + str(conv['layer_idx']))(x)
        if conv['leaky']: x = LeakyReLU(alpha=0.1, name='leaky_' + str(conv['layer_idx']))(x)

    # If the block should have a shortcut, we add the output of the penultimate layer to the output of the last layer.
    return add([skip_connection, x]) if skip else x

# Utility function for applying sigmoid activation to a tensor
def _sigmoid(x):
    return 1. / (1. + np.exp(-x))

# Function for calculating intersection over union (IoU) between two bounding boxes
def bbox_iou(box1, box2):
    # Get the coordinates of the intersection of the two boxes.
    intersect_w = interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])

    # Calculate the area of the intersection.
    intersect = intersect_w * intersect_h

    # Calculate the area of each box
    w1, h1 = box1.xmax - box1.xmin, box1.ymax - box1.ymin
    w2, h2 = box2.xmax - box2.xmin, box2.ymax - box2.ymin

    # Calculate the area of the union of the two boxes.
    union = w1 * h1 + w2 * h2 - intersect

    # The IoU is the ratio of the area of the intersection to the area of the union.
    return float(intersect) / union

# Function for constructing the YOLOv3 model. The function defines the architecture of the model and returns it.
def make_yolov3_model():
    input_image = Input(shape=(None, None, 3))

    #The model consists of several convolutional blocks with skip connections and upsample layers. The architecture is defined in the Darknet-53 config file.
    # Layer  0 => 4
    x = _conv_block(input_image,
                    [{'filter': 32, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 0},
                     {'filter': 64, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 1},
                     {'filter': 32, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 2},
                     {'filter': 64, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 3}])

    # Layer  5 => 8
    x = _conv_block(x, [{'filter': 128, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 5},
                        {'filter': 64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 6},
                        {'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 7}])

    # Layer  9 => 11
    x = _conv_block(x, [{'filter': 64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 9},
                        {'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 10}])

    # Layer 12 => 15
    x = _conv_block(x, [{'filter': 256, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 12},
                        {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 13},
                        {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 14}])

    # Layer 16 => 36
    for i in range(7):
        x = _conv_block(x, [
            {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 16 + i * 3},
            {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 17 + i * 3}])

    skip_36 = x

    # Layer 37 => 40
    x = _conv_block(x, [{'filter': 512, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 37},
                        {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 38},
                        {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 39}])

    # Layer 41 => 61
    for i in range(7):
        x = _conv_block(x, [
            {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 41 + i * 3},
            {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 42 + i * 3}])

    skip_61 = x

    # Layer 62 => 65
    x = _conv_block(x, [{'filter': 1024, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 62},
                        {'filter': 512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 63},
                        {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 64}])

    # Layer 66 => 74
    for i in range(3):
        x = _conv_block(x, [
            {'filter': 512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 66 + i * 3},
            {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 67 + i * 3}])

    # Layer 75 => 79
    x = _conv_block(x, [{'filter': 512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 75},
                        {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 76},
                        {'filter': 512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 77},
                        {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 78},
                        {'filter': 512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 79}],
                    skip=False)

    # Layer 80 => 82
    yolo_82 = _conv_block(x, [{'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 80},
                              {'filter': 255, 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False,
                               'layer_idx': 81}], skip=False)

    # Layer 83 => 86
    x = _conv_block(x, [{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 84}],
                    skip=False)
    x = UpSampling2D(2)(x)
    x = concatenate([x, skip_61])

    # Layer 87 => 91
    x = _conv_block(x, [{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 87},
                        {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 88},
                        {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 89},
                        {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 90},
                        {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 91}],
                    skip=False)

    # Layer 92 => 94
    yolo_94 = _conv_block(x, [{'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 92},
                              {'filter': 255, 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False,
                               'layer_idx': 93}], skip=False)

    # Layer 95 => 98
    x = _conv_block(x, [{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 96}],
                    skip=False)
    x = UpSampling2D(2)(x)
    x = concatenate([x, skip_36])

    # Layer 99 => 106
    yolo_106 = _conv_block(x, [{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 99},
                               {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True,
                                'layer_idx': 100},
                               {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True,
                                'layer_idx': 101},
                               {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True,
                                'layer_idx': 102},
                               {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True,
                                'layer_idx': 103},
                               {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True,
                                'layer_idx': 104},
                               {'filter': 255, 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False,
                                'layer_idx': 105}], skip=False)

    model = Model(input_image, [yolo_82, yolo_94, yolo_106])
    return model


# Preprocess the input image
def preprocess_input(image, net_h, net_w):
    new_h, new_w, _ = image.shape

    # Determine the new size of the image so as to maintain the aspect ratio
    # and make sure it fits within the dimensions specified (net_h, net_w)
    if (float(net_w) / new_w) < (float(net_h) / new_h):
        new_h = (new_h * net_w) / new_w
        new_w = net_w
    else:
        new_w = (new_w * net_h) / new_h
        new_h = net_h

    # Resize the image to the new size and normalize the pixel values
    # cv2.imread loads image in BGR format, converting BGR to RGB
    resized = cv2.resize(image[:, :, ::-1] / 255., (int(new_w), int(new_h)))

    # embed the image into the standard letter box
    new_image = np.ones((net_h, net_w, 3)) * 0.5
    new_image[int((net_h - new_h) // 2):int((net_h + new_h) // 2), int((net_w - new_w) // 2):int((net_w + new_w) // 2),
    :] = resized
    new_image = np.expand_dims(new_image, 0)

    return new_image

# Decode the output of the YOLO network
def decode_netout(netout, anchors, obj_thresh, net_h, net_w):
    grid_h, grid_w = netout.shape[:2]
    nb_box = 3 # Number of bounding boxes per cell

    # Reshape the output to make it easier to process
    netout = netout.reshape((grid_h, grid_w, nb_box, -1))
    nb_class = netout.shape[-1] - 5 # Number of classes is the length of the last dimension minus 5 (for x, y, w, h, and objectness)

    boxes = []

    # Apply sigmoid activation to the x, y coordinates and the objectness score,
    # calculate the class scores
    netout[..., :2] = _sigmoid(netout[..., :2])
    netout[..., 4:] = _sigmoid(netout[..., 4:])
    netout[..., 5:] = netout[..., 4][..., np.newaxis] * netout[..., 5:]
    netout[..., 5:] *= netout[..., 5:] > obj_thresh

    for i in range(grid_h * grid_w):
        row = i / grid_w
        col = i % grid_w

        for b in range(nb_box):
            # 4th element is objectness score
            objectness = netout[int(row)][int(col)][b][4]

            if objectness.all() <= obj_thresh:
                continue

            # first 4 elements are x, y, w, and h
            x, y, w, h = netout[int(row)][int(col)][b][:4]

            # Convert coordinates from relative to grid to relative to image
            x = (col + x) / grid_w  # center position, unit: image width
            y = (row + y) / grid_h  # center position, unit: image height

            # Convert width and height from relative to grid to relative to image
            w = anchors[2 * b + 0] * np.exp(w) / net_w  # unit: image width
            h = anchors[2 * b + 1] * np.exp(h) / net_h  # unit: image height

            # Get the class scores
            classes = netout[int(row)][col][b][5:]

            # Create a new BoundBox instance for every bounding box
            box = BoundBox(x - w / 2, y - h / 2, x + w / 2, y + h / 2, classes)
            if box.xmax == box.xmin or box.ymax == box.ymin:
                continue

            boxes.append(box)

    return boxes

# Adjust bounding boxes to match the size of the original image
def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
    if (float(net_w) / image_w) < (float(net_h) / image_h):
        new_w = net_w
        new_h = (image_h * net_w) / image_w
    else:
        new_h = net_w
        new_w = (image_w * net_h) / image_h

    # Correct the sizes of the bounding boxes
    for i in range(len(boxes)):
        x_offset, x_scale = (net_w - new_w) / 2. / net_w, float(new_w) / net_w
        y_offset, y_scale = (net_h - new_h) / 2. / net_h, float(new_h) / net_h

        boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
        boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
        boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
        boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)

# Apply non-maximum suppression to the detected boxes
def do_nms(boxes, nms_thresh):
    if len(boxes) > 0:
        nb_class = len(boxes[0].classes)
    else:
        return

    # Sort boxes by the score of their most probable class
    sorted_indices = np.argsort([-box.score for box in boxes])

    for i in range(len(sorted_indices)):
        index_i = sorted_indices[i]

        if boxes[index_i].score == 0:
            continue

        for c in range(nb_class):
            if boxes[index_i].classes[c] == 0:
                continue

            for j in range(i + 1, len(sorted_indices)):
                index_j = sorted_indices[j]

                # Suppress the box if it overlaps and belongs to the same class
                if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh and boxes[index_j].classes[c] > 0:
                    boxes[index_j].classes[c] = 0


# This function is used to draw bounding boxes and labels on the image.
def draw_boxes(image, boxes, line, labels, obj_thresh, dcnt):
    font_scale = 1e-3 * image.shape[0]  # font scale for text in image

    # Draw a line on the image
    cv2.line(image, line[0], line[1], (255, 0, 0), 3)

    # Iterate over all detected boxes
    for box in boxes:
        # Get indices of classes with detection probability above threshold
        label_indices = np.where(np.array(box.classes) > obj_thresh)[0]

        # If there are any detections for this bounding box
        if label_indices.size > 0:
            # Prepare label string
            label_str = ''.join([labels[i] for i in label_indices])
            # Coordinates of the rectangle
            (rxmin, rymin) = (box.xmin, box.ymin)
            (rxmax, rymax) = (box.xmax, box.ymax)

            # Get the points of the bounding box
            points = [(rxmin, rymin), (rxmin, rymax), (rxmax, rymin), (rxmax, rymax)]

            tf = False
            # Check if any line from bounding box intersects the predefined line
            for point1, point2 in zip(points, points[1:] + points[:1]):
                if intersection(line[0], line[1], point1, point2):
                    tf = True
                    break

            if tf:
                # Draw the box in red if it intersects with the line
                cv2.rectangle(image, (box.xmin, box.ymin), (box.xmax, box.ymax), (0, 0, 255), 3)

                # Show the violation in a new window
                cimg = image[box.ymin:box.ymax, box.xmin:box.xmax]
                if cimg.shape[0] == 0 or cimg.shape[1] == 0:
                    continue
                cv2.imshow("violation", cimg)
                cv2.waitKey(1)
                dcnt = dcnt + 1
                rect_color = (0, 0, 255)
                thickness = 3

                # Detect number plate in the violation
                number_plate = detect_number_plate(cimg)
                if number_plate is not None:
                    label_str += " (" + number_plate + ")"
                    cv2.imwrite("violations/violation_" + number_plate + ".jpg", cimg)
            else:
                # Draw the box in green if it doesn't intersect with the line
                rect_color = (0, 255, 0)
                thickness = 2

            # Draw bounding box and text
            cv2.rectangle(image, (box.xmin, box.ymin), (box.xmax, box.ymax), rect_color, thickness)

            cv2.putText(image,
                        label_str + ' ' + str(round(box.score, 2)),
                        (box.xmin, box.ymin - 13),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        (0, 255, 255), 2)

    return image

# Path to the trained weights of the YOLOv3 model
weights_path = "resources/object_detection_yolov3.weights"

# set some parameters
net_h, net_w = 416, 416 # dimensions of the input images for the model
obj_thresh, nms_thresh = 0.5, 0.45  # threshold for object detection and non-max suppression
anchors = [[116, 90, 156, 198, 373, 326], [30, 61, 62, 45, 59, 119], [10, 13, 16, 30, 33, 23]] # anchor boxes for the YOLOv3 model

labels = []  # initialize an empty list to store the labels

# Load the labels of the classes that the YOLOv3 model was trained to detect
with open('resources/object_detection_yolov3.labels', 'r') as file:  # open the file in read mode
    for line in file:  # iterate over each line in the file
        labels.append(line.strip())  # strip the newline character at the end and add to the list

#Instantiate the YOLOv3 model, designed to identify 80 distinct classes in the COCO dataset
yolov3 = make_yolov3_model()

# Load the pre-trained weights into the YOLOv3 model
weight_reader = WeightReader(weights_path)
weight_reader.load_weights(yolov3)