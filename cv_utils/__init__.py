from __future__ import division
import pandas as pd
import numpy as np
import cv2
import dlib

CONFIG = None

def read_image(filepath):
  return cv2.imread(filepath)

def write_image(image_data, filepath):
  cv2.imwrite(filepath, image_data)

  return filepath

def convert_to_rgb(image_data, convertionCode=None):
  convertionCode = cv2.COLOR_BGR2RGB if convertionCode is None else convertionCode
  image_data_rgb = cv2.cvtColor(image_data, convertionCode)
  return image_data_rgb

def convert_to_bgr(image_data, convertionCode=None):
  convertionCode = cv2.COLOR_RGB2BGR if convertionCode is None else convertionCode
  image_data_bgr = cv2.cvtColor(image_data, convertionCode)
  
  return image_data_bgr

def convert_to_hsv(image_data, convertionCode=None):
  convertionCode = cv2.COLOR_BGR2HSV if convertionCode is None else convertionCode
  image_data_hsv = cv2.cvtColor(image_data, convertionCode)
  
  return image_data_hsv

def convert_to_gray(image_data, convertionCode=None):
  convertionCode = cv2.COLOR_BGR2GRAY if convertionCode is None else convertionCode
  image_data_gray = cv2.cvtColor(image_data, convertionCode)
  
  return image_data_gray

def convert_bgr_to_redtone(image_data):
  data_tone = np.zeros(image_data.shape)
  data_tone[:,:,2] = image_data[:,:,2]
  return data_tone

def convert_bgr_to_greentone(image_data):
  data_tone = np.zeros(image_data.shape)
  data_tone[:,:,1] = image_data[:,:,1]
  return data_tone

def convert_bgr_to_bluetone(image_data):
  data_tone = np.zeros(image_data.shape)
  data_tone[:,:,0] = image_data[:,:,0]
  return data_tone

def compute_histogram_3channel(image_data):
  result = [None] * 3

  result[0] = cv2.calcHist([image_data], [0], None, [256], [0, 256]).flatten()
  result[1] = cv2.calcHist([image_data], [1], None, [256], [0, 256]).flatten()
  result[2] = cv2.calcHist([image_data], [2], None, [256], [0, 256]).flatten()

  result = np.array(result)

  return result

def compute_histogram_1channel(image_data):
  result = np.bincount(image_data.ravel())
  return result

def compute_downsamples(image_data, n_steps):
  _r = [image_data]
  
  map(lambda v: _r.insert(v+1, cv2.pyrDown(_r[v])), range(n_steps))
  return _r

def compute_upsamples(image_data, n_steps):
  _r = [image_data]
  
  map(lambda v: _r.insert(v+1, cv2.pyrUp(_r[v])), range(n_steps))
  return _r

def coords_pixel_to_local(coords_tuple, image_data):
  shape_image = image_data.shape[:2]
  result = [ x[0]/x[1] for x in zip(coords_tuple, shape_image)]
  result = map(lambda x : max(x,0)[0], zip(result, shape_image))
  
  return tuple(np.clip(result, 0.0, 1.0))

def coords_local_to_pixel(coords_tuple, image_data):
  coords_tuple = tuple(np.clip(coords_tuple, 0, 1))
  shape_image = image_data.shape[:2]

  result = [ x[0]*x[1] for x in zip(coords_tuple, shape_image)]

  return tuple(map(int, result))

def distance_betweencoords(coords_tuple1, coords_tuple2):
  result = np.linalg.norm(np.subtract(coords_tuple2, coords_tuple1))
  return result

def distance_to_origin(coords_tuple2):
  result = np.linalg.norm(np.subtract(coords_tuple2, (0,0)))
  return result

def draw_rectangle(image_data, topl_coord, bottomr_coord, color, thickness=-1):
  ## thickness with negative values means the rectangle needs to be filled
  result = image_data.copy()
  cv2.rectangle(result, topl_coord, bottomr_coord, color, thickness)
  return result

def draw_circle(image_data, center_coord, radius, color, thickness=-1):
  result = image_data.copy()
  cv2.circle(result, center_coord, radius, color, thickness)
  return result

def draw_line(image_data, coord_1, coord_2, color, thickness=0):
  result = image_data.copy()
  cv2.line(result, coord_1, coord_2, color, thickness)
  return result

def draw_polygon(image_data, coords, color, thickness=0, isClosed=True, filled=False):
  if type(coords) == list:
    coords = np.array(coords)

  coords = coords.reshape((-1,1,2))

  result = image_data.copy()
  
  if filled:
    cv2.fillPoly(result, [coords], color)
  else:
    cv2.polylines(result, [coords], isClosed, color, thickness)

  return result

def translate_inpixels(image_data, x_move, y_move):
  result = None

  # T is our translation matrix
  T = np.float32([[1, 0, x_move], [0, 1,y_move]])

  # We use warpAffine to transform the image using the matrix, T
  result = cv2.warpAffine(image_data, T, image_data.shape[:2])

  return result

def translate_inlocal(image_data, x_move, y_move):
  height, width = image_data.shape[:2]
  return translate_inpixels(image_data, x_move*width, y_move*height)

def crop_inpixels(image_data, topl_coord, bottomr_coord):
  result = image_data[topl_coord[1]:bottomr_coord[1], topl_coord[0]:bottomr_coord[0]]
  return result

def crop_inlocal(image_data, topl_coord, bottomr_coord):
  height, width = image_data.shape[:2]

  topl_coord = [topl_coord[0] * width, topl_coord[1]*height]
  bottomr_coord = [bottomr_coord[0] * width, bottomr_coord[1]*height]

  topl_coord = tuple(map(int, topl_coord))
  bottomr_coord = tuple(map(int, bottomr_coord))


  return crop_inpixels(image_data, topl_coord, bottomr_coord)

def rotate(image_data, angle, pivot_coords=None, scale=0.5):
  height, width = image_data.shape[:2]

  if pivot_coords is None:
    pivot_coords = (height/2, width/2)

  rotation_matrix = cv2.getRotationMatrix2D(pivot_coords, angle, scale)

  rotated_image = cv2.warpAffine(image_data, rotation_matrix, (width, height))

  return rotated_image

def flip(image_data, axis):
  flipped = cv2.flip(image_data, axis)
  return flipped

def scale_inpixels(image_data, x_target, y_target):
  height, width = image_data.shape[:2]
  return scale_inlocal(image_data, (x_target/width), (y_target/height))

def scale_inlocal(image_data, x_factor, y_factor):
  result = cv2.resize(image_data, None, fx=x_factor, fy=y_factor)
  return result

def bit_and(mask1, mask2):
  result = cv2.bitwise_and(mask1, mask2)
  return result

def bit_or(mask1, mask2):
  result = cv2.bitwise_or(mask1, mask2)
  return result

def bit_xor(mask1, mask2):
  result = cv2.bitwise_xor(mask1, mask2)
  return result

def bit_not(mask1):
  result = cv2.bitwise_not(mask1)
  return result

def apply_mask(image_data, mask):
  result = cv2.bitwise_and(image_data, image_data, mask = mask)
  return result

def blur_box(image_data, kernel):
  if type(kernel) is int:
    kernel = (kernel, kernel)

  result = cv2.blur(image_data, kernel)
  return result

def blur_gausian(image_data, kernel):
  if type(kernel) is int:
    kernel = (kernel, kernel)

  result = cv2.GaussianBlur(image_data, kernel, 0)
  return result

def blur_median(image_data, kernel):
  result = cv2.medianBlur(image_data, kernel)
  return result

def blur_bilateral(image_data, d=9, sigmaColor=75, sigmaSpace=75):
  result = cv2.bilateralFilter(image_data, d, sigmaColor, sigmaSpace)
  return result

def denoise(image_data, luminanceStrenght=6, window_size=7, search_window_size=21):
  result = cv2.fastNlMeansDenoisingColored(image_data, luminanceStrenght, luminanceStrenght, 6, window_size, search_window_size)

  return result

def sharpen(image_data):
  kernel_sharpening = np.array([[-1,-1,-1], 
                                [-1,9,-1], 
                                [-1,-1,-1]])

  # applying different kernels to the input image
  result = cv2.filter2D(image_data, -1, kernel_sharpening)

  return result

def threshold_binary(image_data, thresoldAt, thresoldTo=255):
  _, result = cv2.threshold(image_data, thresoldAt, thresoldTo, cv2.THRESH_BINARY)
  return result

def threshold_binary_inverse(image_data, thresoldAt, thresoldTo=255):
  _, result = cv2.threshold(image_data, thresoldAt, thresoldTo, cv2.THRESH_BINARY_INV)
  return result

def threshold_truncate(image_data, thresoldAt, thresoldTo=255):
  _, result = cv2.threshold(image_data, thresoldAt, thresoldTo, cv2.THRESH_TRUNC)
  return result

def threshold_tozero(image_data, thresoldAt, thresoldTo=255):
  _, result = cv2.threshold(image_data, thresoldAt, thresoldTo, cv2.THRESH_TOZERO)
  return result

def threshold_tozero_inverse(image_data, thresoldAt, thresoldTo=255):
  _, result = cv2.threshold(image_data, thresoldAt, thresoldTo, cv2.THRESH_TOZERO_INV)
  return result

def threshold_adaptive(image_data, thresoldTo=255, blocksize=3, C=5):
  """
    .   @param blockSize Size of a pixel neighborhood that is used to calculate a threshold value for the
    .   pixel: 3, 5, 7, and so on.
    .   @param C Constant subtracted from the mean or weighted mean (see the details below). Normally, it
  """
  result = cv2.adaptiveThreshold(image_data, thresoldTo, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY, blocksize, C)

  return result

def threshold_otsu(image_data, thresoldAt, thresoldTo=255):
  _, result = cv2.threshold(image_data, thresoldAt, thresoldTo, 
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)

  return result

def threshold_gaussianotsu(image_data, thresoldAt, thresoldTo=255):
  blur = cv2.GaussianBlur(image_data, (3,3), 0)
  _, result = cv2.threshold(image_data, thresoldAt, thresoldTo, 
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)

  return result

def morph_dilate(image_data, kernel):
  if type(kernel) is int:
    kernel = (kernel, kernel)

  kernel = np.ones(kernel, np.uint8)

  result = cv2.dilate(image_data, kernel, iterations = 1)
  return result

def morph_erode(image_data, kernel):
  if type(kernel) is int:
    kernel = (kernel, kernel)

  kernel = np.ones(kernel, np.uint8)

  result = cv2.erode(image_data, kernel, iterations = 1)
  return result

def morph_open(image_data, kernel):
  if type(kernel) is int:
    kernel = (kernel, kernel)

  kernel = np.ones(kernel, np.uint8)

  result = cv2.morphologyEx(image_data, cv2.MORPH_OPEN, kernel)
  return result

def morph_close(image_data, kernel):
  if type(kernel) is int:
    kernel = (kernel, kernel)

  kernel = np.ones(kernel, np.uint8)

  result = cv2.morphologyEx(image_data, cv2.MORPH_CLOSE, kernel)
  return result

def edge_sobel_x(image_data, kernel_size):
  if image_data.ndim > 2:
    image_data = convert_to_gray(image_data)

  result = cv2.Sobel(image_data, cv2.CV_64F, 0, 1, ksize=kernel_size)
  return result

def edge_sobel_y(image_data, kernel_size):
  if image_data.ndim > 2:
    image_data = convert_to_gray(image_data)

  result = cv2.Sobel(image_data, cv2.CV_64F, 1, 0, ksize=kernel_size)
  return result

def edge_laplacian(image_data):
  if image_data.ndim > 2:
    image_data = convert_to_gray(image_data)

  result = cv2.Laplacian(image_data, cv2.CV_64F)
  return result

def edge_canny(image_data, threshold1, threshold2):
  if image_data.ndim > 2:
    image_data = convert_to_gray(image_data)

  result = cv2.Canny(image_data, threshold1, threshold2)
  return result

def edge_canny_blur(image_data, threshold1, threshold2):
  if image_data.ndim > 2:
    image_data = convert_to_gray(image_data)

  image_data = cv2.blur(image_data, (5,5))

  result = cv2.Canny(image_data, threshold1, threshold2)
  return result

def perspective_affine(image_data, coords_inzorder, coords2_inzorder):
  rows,cols,ch = image_data.shape

  points_A = np.float32(coords_inzorder)
  points_B = np.float32(coords2_inzorder)
  M = cv2.getAffineTransform(points_A, points_B)
  result = cv2.warpAffine(image_data, M, (cols, rows))
  return result

def perspective_nonaffine(image_data, coords_inzorder, coords2_inzorder, target_size=None):
  target_size = image_data.shape[:2] if target_size is None else target_size
  points_A = np.float32(coords_inzorder)
  points_B = np.float32(coords2_inzorder)
  transformation_matrix = cv2.getPerspectiveTransform(points_A, points_B)

  result = cv2.warpPerspective(image_data, transformation_matrix, target_size)
  return result

def compute_contours(image_data, threshold1=0, threshold2=255, canny_threshold1=30, canny_threshold2=200):
  if image_data.ndim > 2:
    image_data = convert_to_gray(image_data)

  ret, threshold = cv2.threshold(image_data, threshold1, threshold2, cv2.THRESH_BINARY_INV)
  edged = cv2.Canny(threshold, canny_threshold1, canny_threshold2)
  _, contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

  return contours, hierarchy

def contour_centroid(contour):
  M = cv2.moments(contour)
  cX = int(M["m10"] / M["m00"])
  cY = int(M["m01"] / M["m00"])
  return (cX, cY)

def contour_area(contour):
  area = cv2.contourArea(contour)
  return area

def contour_boundingbox(contour):
  x,y,w,h = cv2.boundingRect(contour)
  result = (x,y) , (x+w,y+h)
  return result

def contours_highlight(image_data, contours_list, color=(0,0,255), thickness=3):
  result = image_data.copy()
  result = cv2.drawContours(result, contours_list, -1, color, thickness)

  return result

def contour_approximation(contour, accuracy=0.03):
  accuracy = accuracy * cv2.arcLength(contour, True)
  approx = cv2.approxPolyDP(contour, accuracy, True)
  return approx

def contour_hull_approximation(contour):
  result = cv2.convexHull(contour)
  return result

def match_shape(contour, template_contour, method=cv2.CONTOURS_MATCH_I3):
  result = cv2.matchShapes(template_contour, contour, method, 0.0)
  return result

def match_hough_lines(edges_data, threshold, rho=1, theta=(np.pi/180)):
  result = []
  hlines = cv2.HoughLines(edges_data, rho, theta, threshold)
  
  if hlines is not None:
    for [[rho,theta]] in hlines:
      a = np.cos(theta)
      b = np.sin(theta)
      x0 = a*rho
      y0 = b*rho
      x1 = int(x0 + 1000*(-b))
      y1 = int(y0 + 1000*(a))
      x2 = int(x0 - 1000*(-b))
      y2 = int(y0 - 1000*(a))

      result.append( [(x1,y1),(x2,y2)] )

  return result

def match_prob_hough_lines(edges_data, threshold, minLineLength, maxLineGap,rho=1, theta=(np.pi/180)):
  hlines = cv2.HoughLinesP(edges_data, rho, theta, threshold, minLineLength=minLineLength, maxLineGap=maxLineGap  )
  result = []

  if hlines is not None:
    for [[x1, y1, x2, y2]] in hlines:
      result.append([(x1, y1), (x2, y2)])

  return result

def match_circle(edges_data, min_dist, dp=1.5, method=cv2.HOUGH_GRADIENT):
  circles = cv2.HoughCircles(edges_data, method, dp, min_dist)

  circles = np.uint16(np.around(circles))

  centers = []
  radius = []

  if circles is not None:
    for i in circles[0,:]:
      # draw the outer circle
      centers.append((i[0], i[1]))
      
      # draw the center of the circle
      radius.append((i[2]))

  return centers, radius

def match_blobs(image_data):
  centers = []
  radius = []
  detector = cv2.SimpleBlobDetector_create()
  keypoints = detector.detect(image_data)

  if keypoints is not None:
    centers = [ x.pt for x in keypoints ]
    radius = [ x.size for x in keypoints ]

  return centers, radius, keypoints

def describe_corners(image_data, blocksize, ksize, k):
  if image_data.ndim > 2:
    image_data = convert_to_gray(image_data)

  image_data = np.float32(image_data)

  harris_corners = cv2.cornerHarris(image_data, blocksize, ksize, k)

  return harris_corners

def describe_good2track(image_data, distBetweenCorners, qualityLevel=0.01, maxCorners=100):
  if image_data.ndim > 2:
    image_data = convert_to_gray(image_data)

  corners = cv2.goodFeaturesToTrack(image_data, maxCorners, qualityLevel, distBetweenCorners)

  return np.squeeze(corners, 1)

def describe_FAST(image_data):
  centers = []
  radius = []

  if image_data.ndim > 2:
    image_data = convert_to_gray(image_data)

  fast = cv2.FastFeatureDetector_create()
  keypoints = fast.detect(image_data, None)

  if keypoints is not None:
    centers = [ x.pt for x in keypoints ]
    radius = [ x.size for x in keypoints ]

  return centers, radius, keypoints

def describe_BRIEF(image_data):
  centers = []
  radius = []

  if image_data.ndim > 2:
    image_data = convert_to_gray(image_data)

  fast = cv2.xfeatures2d.StarDetector_create()
  brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
  keypoints = fast.detect(image_data, None)
  keypoints, descriptors = brief.compute(image_data, keypoints)

  if keypoints is not None:
    centers = [ x.pt for x in keypoints ]
    radius = [ x.size for x in keypoints ]

  return centers, radius, keypoints

def describe_ORB(image_data):
  centers = []
  radius = []

  if image_data.ndim > 2:
    image_data = convert_to_gray(image_data)

  orb = cv2.ORB_create()
  keypoints = orb.detect(image_data, None)

  keypoints, descriptors = orb.compute(image_data, keypoints)

  if keypoints is not None:
    centers = [ x.pt for x in keypoints ]
    radius = [ x.size for x in keypoints ]

  return centers, radius, keypoints

def describe_HOG(image_data, cellsize=(8,8), blocksize=(2,2), nbins=9):
  if image_data.ndim > 2:
    image_data = convert_to_gray(image_data)

  # Using OpenCV's HOG Descriptor
  # winSize is the size of the image cropped to a multiple of the cell size
  hog = cv2.HOGDescriptor(_winSize=(image_data.shape[1] // cellsize[1] * cellsize[1],
                                    image_data.shape[0] // cellsize[0] * cellsize[0]),
                          _blockSize=(blocksize[1] * cellsize[1],
                                      blocksize[0] * cellsize[0]),
                          _blockStride=(cellsize[1], cellsize[0]),
                          _cellSize=(cellsize[1], cellsize[0]),
                          _nbins=nbins)

  # Create numpy array shape which we use to create hog_feats
  n_cells = (image_data.shape[0] // cellsize[0], image_data.shape[1] // cellsize[1])

  # We index blocks by rows first.
  # hog_feats now contains the gradient amplitudes for each direction,
  # for each cell of its group for each group. Indexing is by rows then columns.
  hog_feats = hog.compute(image_data).reshape(n_cells[1] - blocksize[1] + 1,
                          n_cells[0] - blocksize[0] + 1,
                          blocksize[0], blocksize[1], nbins).transpose((1, 0, 2, 3, 4))

  # Create our gradients array with nbin dimensions to store gradient orientations 
  gradients = np.zeros((n_cells[0], n_cells[1], nbins))

  # Create array of dimensions 
  cell_count = np.full((n_cells[0], n_cells[1], 1), 0, dtype=int)

  # Block Normalization
  for off_y in range(blocksize[0]):
      for off_x in range(blocksize[1]):
          gradients[off_y:n_cells[0] - blocksize[0] + off_y + 1,
                    off_x:n_cells[1] - blocksize[1] + off_x + 1] += \
              hog_feats[:, :, off_y, off_x, :]
          cell_count[off_y:n_cells[0] - blocksize[0] + off_y + 1,
                     off_x:n_cells[1] - blocksize[1] + off_x + 1] += 1

  # Average gradients
  gradients /= cell_count

  return gradients, hog_feats

def match_face(image_data, scaleFactor=1.3, minNeighbors=5):
  face_classifier = cv2.CascadeClassifier(CONFIG["HAARCASCADE_FRONTALFACE_DEFAULT"])

  if image_data.ndim > 2:
    image_data = convert_to_gray(image_data)

  faces = face_classifier.detectMultiScale(image_data, scaleFactor, minNeighbors)

  if faces is ():
      faces = []

  faces = [[(x,y), (x+w,y+h)] for x,y,w,h in faces]

  return faces

def match_eyes(image_data, scaleFactor=1.1, minNeighbors=5):
  eye_classifier = cv2.CascadeClassifier(CONFIG["HAARCASCADE_EYE"])
  if image_data.ndim > 2:
    image_data = convert_to_gray(image_data)

  eyes = eye_classifier.detectMultiScale(image_data, scaleFactor, minNeighbors)
  
  if eyes is ():
      eyes = []

  eyes = [[(x,y), (x+w,y+h)] for x,y,w,h in eyes]

  return eyes

def match_facial_landmarks(image_data):
  result = []
  predictor = dlib.shape_predictor(CONFIG["SHAPE_PREDICTOR_68_FACE_LANDMARKS"])
  detector = dlib.get_frontal_face_detector()

  rects = detector(image_data, 1)

  for rect in rects:
    result.append(np.matrix([[p.x, p.y] for p in predictor(image_data, rect).parts()]))

  return result

def compute_faceswap():
  pass

def filter_by_color():
  pass

def __set_config(config=None):
  global CONFIG

  if config is None:
    config = dict()
    config["HAARCASCADE_EYE"] = "haarcascade_eye.xml"
    config["HAARCASCADE_FRONTALFACE_DEFAULT"] = "haarcascade_frontalface_default.xml"
    config["SHAPE_PREDICTOR_68_FACE_LANDMARKS"] = "shape_predictor_68_face_landmarks.dat"

  CONFIG = config

  return CONFIG

__set_config()














