import argparse
import sys
import time
from matplotlib import pyplot as plt
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
import tensorflow_hub as hub
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from utils import visualize


def run(model: str, camera_id: int, width: int, height: int) -> None:
  """Continuously run inference on images acquired from the camera.

  Args:
    model: Name of the TFLite object detection model.
    camera_id: The camera id to be passed to OpenCV.
    width: The width of the frame captured from the camera.
    height: The height of the frame captured from the camera.
  """

  # Variables to calculate FPS
  counter, fps = 0, 0
  start_time = time.time()

  # Start capturing video input from the camera
  cap = cv2.VideoCapture("tp2.mp4")
  original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
  original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

  # Visualization parameters
  row_size = 20  # pixels
  left_margin = 24  # pixels
  text_color = (0, 0, 255)  # red
  font_size = 1
  font_thickness = 1
  fps_avg_frame_count = 10
  
  detection_result_list = []

  def visualize_callback(result: vision.ObjectDetectorResult,
                         output_image: mp.Image, timestamp_ms: int):
      result.timestamp_ms = timestamp_ms
      detection_result_list.append(result)


  # Initialize the object detection model
  base_options = python.BaseOptions(model_asset_path=model)
  options = vision.ObjectDetectorOptions(base_options=base_options,
                                         running_mode=vision.RunningMode.VIDEO,
                                         score_threshold=0.28,
                                         category_allowlist=['person'])

  detector = vision.ObjectDetector.create_from_options(options)

  gpus = tf.config.experimental.list_physical_devices('GPU')
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
      
#   model = hub.load('https://tfhub.dev/google/movenet/multipose/lightning/1')
  model = hub.load("trainPosture")
  print("\n\n",model,"\n\n")
  movenet = model.signatures['serving_default']
  
  # Continuously capture images from the camera and run inference
  while cap.isOpened():
    success, image = cap.read()
    image = cv2.resize(image, (1280, 720))
    # resize_window('Video', width, height)
    
    img = image.copy()
    img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 384,640)
    input_img = tf.cast(img, dtype=tf.int32)
    
    # Detection section
    results = movenet(input_img)
    keypoints_with_scores = results['output_0'].numpy()[:,:,:51].reshape((6,17,3))
    
    # Render keypoints 
    loop_through_people(image, keypoints_with_scores, EDGES, 0.25)
    
    
    
    if not success:
      sys.exit(
          'ERROR: Unable to read from webcam. Please verify your webcam settings.'
      )

    counter += 1
    image = cv2.flip(image, 1)

    # Convert the image from BGR to RGB as required by the TFLite model.
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

    # Run object detection using the model.
    detection_result = detector.detect_for_video(mp_image, counter)
    detection_result_list.append(detection_result)
    current_frame = mp_image.numpy_view()
    current_frame = cv2.cvtColor(current_frame, cv2.COLOR_RGB2BGR)

    # Calculate the FPS
    if counter % fps_avg_frame_count == 0:
        end_time = time.time()
        fps = fps_avg_frame_count / (end_time - start_time)
        start_time = time.time()

    # Show the FPS
    fps_text = 'FPS = {:.1f}'.format(fps)
    text_location = (left_margin, row_size)
    cv2.putText(current_frame, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                font_size, text_color, font_thickness)

    if detection_result_list:
        # print(detection_result_list)
        vis_image = visualize(current_frame, detection_result_list[0])
        cv2.imshow('object_detector', vis_image)
        detection_result_list.clear()
    else:
        cv2.imshow('object_detector', current_frame)

    # Stop the program if the ESC key is pressed.
    if cv2.waitKey(1) == 27:
      break

  detector.close()
  cap.release()
  cv2.destroyAllWindows()

def loop_through_people(frame, keypoints_with_scores, edges, confidence_threshold):
    for person in keypoints_with_scores:
        draw_connections(frame, person, edges, confidence_threshold)
        draw_keypoints(frame, person, confidence_threshold)
        
def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 6, (0,255,0), -1)
            

EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):      
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 4)





def main():
  
  
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model',
      help='Path of the object detection model.',
      required=False,
      default='efficientdet.tflite')
  parser.add_argument(
      '--cameraId', help='Id of camera.', required=False, type=int, default=0)
  parser.add_argument(
      '--frameWidth',
      help='Width of frame to capture from camera.',
      required=False,
      type=int,
      default=1280)
  parser.add_argument(
      '--frameHeight',
      help='Height of frame to capture from camera.',
      required=False,
      type=int,
      default=720)
  args = parser.parse_args()

  run(args.model, int(args.cameraId), args.frameWidth, args.frameHeight)


if __name__ == '__main__':
  main()



# import argparse
# import sys
# import time

# import cv2
# import mediapipe as mp

# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision

# from utils import visualize


# def run(model: str, camera_id: int, width: int, height: int) -> None:
#   """Continuously run inference on images acquired from the camera.

#   Args:
#     model: Name of the TFLite object detection model.
#     camera_id: The camera id to be passed to OpenCV.
#     width: The width of the frame captured from the camera.
#     height: The height of the frame captured from the camera.
#   """

#   # Variables to calculate FPS
#   counter, fps = 0, 0
#   start_time = time.time()

#   # Start capturing video input from the camera
#   cap = cv2.VideoCapture('testdemo.mp4')
#   cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
#   cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

#   # Visualization parameters
#   row_size = 20  # pixels
#   left_margin = 24  # pixels
#   text_color = (0, 0, 255)  # red
#   font_size = 1
#   font_thickness = 1
#   fps_avg_frame_count = 10

#   detection_result_list = []

#   def visualize_callback(result: vision.ObjectDetectorResult,
#                          output_image: mp.Image, timestamp_ms: int):
#       result.timestamp_ms = timestamp_ms
#       detection_result_list.append(result)


#   # Initialize the object detection model
#   base_options = python.BaseOptions(model_asset_path=model)
#   options = vision.ObjectDetectorOptions(base_options=base_options,
#                                          running_mode=vision.RunningMode.VIDEO,
#                                          score_threshold=0.22,
#                                          category_allowlist=['person'])
  
#   detector = vision.ObjectDetector.create_from_options(options)


#   # Continuously capture images from the camera and run inference
#   while cap.isOpened():
#     success, image = cap.read()
#     image = cv2.resize(image,(1200,700))
#     if not success:
#       sys.exit(
#           'ERROR: Unable to read from webcam. Please verify your webcam settings.'
#       )

#     counter += 1
#     image = cv2.flip(image, 1)

#     # Convert the image from BGR to RGB as required by the TFLite model.
#     rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

#     # Run object detection using the model.
#     detection_result = detector.detect_for_video(mp_image, counter)
#     detection_result_list.append(detection_result)
#     current_frame = mp_image.numpy_view()
#     current_frame = cv2.cvtColor(current_frame, cv2.COLOR_RGB2BGR)

#     # Calculate the FPS
#     if counter % fps_avg_frame_count == 0:
#         end_time = time.time()
#         fps = fps_avg_frame_count / (end_time - start_time)
#         start_time = time.time()

#     # Show the FPS
#     fps_text = 'FPS = {:.1f}'.format(fps)
#     text_location = (left_margin, row_size)
#     cv2.putText(current_frame, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
#                 font_size, text_color, font_thickness)

#     if detection_result_list:
#         print(detection_result_list)
#         vis_image = visualize(current_frame, detection_result_list[0])
#         cv2.imshow('object_detector', vis_image)
#         detection_result_list.clear()
#     else:
#         cv2.imshow('object_detector', current_frame)

#     # Stop the program if the ESC key is pressed.
#     if cv2.waitKey(1) == 27:
#       break

#   detector.close()
#   cap.release()
#   cv2.destroyAllWindows()


# def main():
#   parser = argparse.ArgumentParser(
#       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#   parser.add_argument(
#       '--model',
#       help='Path of the object detection model.',
#       required=False,
#       default='efficientdet.tflite')
#   parser.add_argument(
#       '--cameraId', help='Id of camera.', required=False, type=int, default=0)
#   parser.add_argument(
#       '--frameWidth',
#       help='Width of frame to capture from camera.',
#       required=False,
#       type=int,
#       default=1280)
#   parser.add_argument(
#       '--frameHeight',
#       help='Height of frame to capture from camera.',
#       required=False,
#       type=int,
#       default=720)
#   args = parser.parse_args()

#   run(args.model, int(args.cameraId), args.frameWidth, args.frameHeight)


# if __name__ == '__main__':
#   main()