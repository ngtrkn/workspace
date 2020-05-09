import sys
import argparse
from yolo import YOLO, detect_video
from PIL import Image
import numpy as np
import cv2
import os

def preprocessing(img_cv):
    # to rgb
    rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    print(rgb.shape)
    # to gray
    gray = np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
    img = np.dstack((gray for i in range(3)))
    return Image.fromarray(img.astype(np.uint8))


def imread(img_path, mode=cv2.IMREAD_COLOR):
    stream = open(img_path, "rb")
    bytes = bytearray(stream.read())
    numpyarray = np.asarray(bytes, dtype=np.uint8)
    bgrImage = cv2.imdecode(numpyarray, mode)
    return bgrImage

def detect_img(yolo, image_path):
    #img = input('Input image filename:')
    try:
        image = preprocessing(imread(image_path))
    except:
        print('Open Error! Try again!')
        
    else:
        r_image = yolo.detect_image(image)
        r_image.save("demo.png")
    yolo.close_session()

FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default='./path2your_video',
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )

    FLAGS = parser.parse_args()


    # SOTA option
    SOTA_opt = {
        "model_path": 'model_data/pretrain/logsep039-loss29.589-val_loss156.046.SOTA.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/custom_object_SOTA.txt',
        "score" : 0.05,
        "iou" : 0.45,
        "model_image_size" : (128, 4096),
        "gpu_num" : 0,
        "color": (128,128,128),
    }
    #FLAGS.update(SOTA_opt)
    image_path = "/mnt/sda1/data/cinnamon/SOTA/test/2/2_6_3028.png"
    detect_img(YOLO(**SOTA_opt), image_path)

    """
    if FLAGS.image:
        
        # Image detection mode, disregard any remaining command line arguments
        
        print("Image detection mode")
        if "input" in FLAGS:
            print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
        detect_img(YOLO(**vars(FLAGS)))
    elif "input" in FLAGS:
        detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")
    """
