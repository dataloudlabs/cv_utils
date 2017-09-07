from __future__ import division
import pandas as pd
import numpy as np
import cv2


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
  shape_image = image_data.shape[:-1]
  result = [ x[0]/x[1] for x in zip(coords_tuple, shape_image)]
  result = map(lambda x : max(x,0)[0], zip(result, shape_image))
  
  return tuple(np.clip(result, 0.0, 1.0))

def coords_local_to_pixel(coords_tuple, image_data):
  coords_tuple = tuple(np.clip(coords_tuple, 0, 1))
  shape_image = image_data.shape[:-1]

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

def draw_polygon(image_data, coords, color, thickness=0, isClosed=True):
  if type(coords) == list:
    coords = np.array(coords)

  coords = coords.reshape((-1,1,2))

  result = image_data.copy()
  cv2.polylines(result, [coords], isClosed, color, thickness)

  return result

def translate_inpixels(x_move, y_move, image_data):
  pass

def translate_inlocal(x_move, y_move, image_data):
  pass

def crop_inpixels(top, bottom, left, right, image_data):
  pass

def crop_inlocal(top, bottom, left, right, image_data):
  pass

def rotate(angle, pivot_coords, image_data):
  pass

def flip(image_data, axis):
  pass

def scale_inpixels(x_target, y_target, image_data):
  pass

def scale_inlocal(x_factor, y_factor, image_data):
  pass

def bit_and(mask1, mask2):
  pass

def bit_or(mask1, mask2):
  pass

def bit_xor(mask1, mask2):
  pass

def bit_not(mask1):
  pass

def apply_mask(image_data, mask):
  pass

def blur(image_data, kernel):
  pass

def blur_kernel(image_data, kernel):
  pass

def blur_gausian(image_data, kernel):
  pass

def blur_median(image_data):
  pass

def blur_bilateral(image_data):
  pass

def denoise(image_data):
  pass

def sharpen(image_data):
  pass

def threshold_binary(image_data, threshold1, threshold2):
  pass

def threshold_binary_inverse(image_data, threshold1, threshold2):
  pass

def threshold_truncate(image_data, threshold1, threshold2):
  pass

def threshold_tozero(image_data, threshold1, threshold2):
  pass

def threshold_tozero_inverse(image_data, threshold1, threshold2):
  pass

def threshold_adaptive(image_data, threshold1, threshold2):
  pass

def threshold_otsu(image_data, threshold1, threshold2):
  pass

def morph_dilate(image_data, kernel):
  pass

def morph_erode(image_data, kernel):
  pass

def morph_open(image_data, kernel):
  pass

def morph_close(image_data, kernel):
  pass

def edge_sobel_x(image_data, kernel_size):
  pass

def edge_sobel_y(image_data, kernel_size):
  pass

def edge_laplacian(image_data):
  pass

def edge_canny(image_data, threshold1, threshold2):
  pass

def edge_canny_blur(image_data, threshold1, threshold2, kernel):
  pass

def perspective_transform():
  pass

def compute_contours(image_data, threshold1, threshold2, canny_threshold1, canny_threshold2):
  pass

def contours_sort_left_to_right(image_data, contours):
  pass

def contours_sort_top_down(image_data, contours):
  pass

def contours_sort_from_origin(image_data, contours):
  pass

def contours_sort_by_area(image_data, contours):
  pass

def contours_centroid(contours):
  pass

def contours_highlight(image_data, contours):
  pass

def contour_approximation(contour):
  pass

def contour_hull_approximation(contour):
  pass

def match_shape(image_data, template):
  pass

def match_hough_lines(image_data, canny_threshold1, canny_threshold2, value):
  pass

def match_prob_hough_lines(image_data, canny_threshold1, canny_threshold2, value, p1, p2):
  pass

def match_circle():
  pass

def match_blobs():
  pass

def describe_corners():
  pass

def describe_good2track():
  pass

def describe_FAST():
  pass

def describe_BRIEF():
  pass

def describe_ORG():
  pass

def describe_HOG():
  pass

def match_face():
  pass

def match_eyes():
  pass

def match_facial_landmarks():
  pass

def compute_faceswap():
  pass

def filter_by_color():
  pass


















