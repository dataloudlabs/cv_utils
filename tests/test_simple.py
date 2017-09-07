import cv_utils
import pickle
import numpy as np

TEST_DATA = pickle.load(open( "./tests/test_data.p", "rb" ))


def helper_listtestdata():
  result = test_simple.TEST_DATA.keys()
  return result

def helper_dummychannels():
  data = np.zeros((256,256,3))
  # vertical gradiant from 0 to 255
  data[:,:,0] = np.repeat(np.expand_dims(np.arange(0, 256, 1), 1), 256, 1)
  #horizontal gradient from 0 to 255
  data[:,:,1] = data[:,:,0].T
  #all 255
  data[:,:,2] = np.ones((256,256)) * 255
  
  data = data.astype(np.uint8)

  return data

def helper_getimage(filename):
  result = TEST_DATA["resources/{}".format(filename)]
  return result

def helper_writetoTEST_DATA(key, value):
  TEST_DATA[key] = value
  pickle.dump(TEST_DATA, open( "./tests/test_data.p", "wb" ))

def test_convert_to_rgb():
  image_data = helper_getimage("cake.jpg")
  expected = helper_getimage("cake_rgb.jpg")
  result = cv_utils.convert_to_rgb(image_data)

  assert np.all(np.equal(expected, result))

def test_convert_to_bgr():
  image_data = helper_getimage("cake_rgb.jpg")
  expected = helper_getimage("cake.jpg")
  result = cv_utils.convert_to_bgr(image_data)

  assert np.all(np.equal(expected, result))

def test_convert_to_gray():
  image_data = helper_getimage("cake.jpg")
  expected = helper_getimage("cake_gray.jpg")
  
  result = cv_utils.convert_to_gray(image_data)

  assert np.all(np.equal(expected, result))

def test_convert_bgr_to_redtone():
  image_data = helper_getimage("cake.jpg")
  expected = np.zeros((612, 816, 3))
  expected[:,:,2] = image_data[:,:,2]
  result = cv_utils.convert_bgr_to_redtone(image_data)
  assert np.all(np.equal(result, expected))

def test_convert_bgr_to_greentone():
  image_data = helper_getimage("cake.jpg")
  expected = np.zeros((612, 816, 3))
  expected[:,:,1] = image_data[:,:,1]
  result = cv_utils.convert_bgr_to_greentone(image_data)
  assert np.all(np.equal(result, expected))

def test_convert_bgr_to_bluetone():
  image_data = helper_getimage("cake.jpg")
  expected = np.zeros((612, 816, 3))
  expected[:,:,0] = image_data[:,:,0]
  result = cv_utils.convert_bgr_to_bluetone(image_data)
  assert np.all(np.equal(result, expected))

def test_compute_histogram_3channel():
  dummy_data = helper_dummychannels()
  result = cv_utils.compute_histogram_3channel(dummy_data)

  expected = [None] * 3
  expected[0] = np.repeat(256, 256, 0)
  expected[1] = np.repeat(256, 256, 0)
  expected[2] = np.zeros((256))
  expected[2][-1] = 65536 # 256*256

  assert np.all(np.equal(result[0], expected[0]))
  assert np.all(np.equal(result[1], expected[1]))
  assert np.all(np.equal(result[2], expected[2]))

def test_compute_histogram_1channel():
  dummy_data = helper_dummychannels()
  result = cv_utils.compute_histogram_1channel(dummy_data)

  expected = np.repeat(512, 256, 0)
  expected[-1] = 66048 #(256*256)+512

  assert np.all(np.equal(result, expected))

def test_compute_downsamples():
  dummy_data = helper_dummychannels()
  exp_shapes = [(256, 256, 3),
              (128, 128, 3),
              (64, 64, 3),
              (32, 32, 3),
              (16, 16, 3),
              (8, 8, 3),
              (4, 4, 3),
              (2, 2, 3),
              (1, 1, 3),
              (1, 1, 3)]


  result = cv_utils.compute_downsamples(dummy_data, 8)
  assert len(result) == 9
  assert np.all(map(lambda x : exp_shapes[x] == result[x].shape, range(9)))

  # this one goes 1 step over the downsampling limit for our example 256x256 image
  result = cv_utils.compute_downsamples(dummy_data, 9)
  assert len(result) == 10
  assert np.all(map(lambda x : exp_shapes[x] == result[x].shape, range(10)))

  result = cv_utils.compute_downsamples(dummy_data, 1)
  assert len(result) == 2
  assert np.all(map(lambda x : exp_shapes[x] == result[x].shape, range(2)))

  result = cv_utils.compute_downsamples(dummy_data, 0)
  assert len(result) == 1
  assert np.all(map(lambda x : exp_shapes[x] == result[x].shape, range(1)))

def test_compute_upsamples():
  dummy_data = helper_dummychannels()
  exp_shapes = [(256, 256, 3), 
                (512, 512, 3), 
                (1024, 1024, 3)]


  result = cv_utils.compute_upsamples(dummy_data, 2)
  assert len(result) == 3
  assert np.all(map(lambda x : exp_shapes[x] == result[x].shape, range(3)))

def test_coords_pixel_to_local():
  dummy_data = helper_dummychannels()
  assert (0.1, 0.2) == cv_utils.coords_pixel_to_local((25.6,51.2), dummy_data)
  assert (0.0, 0.2) == cv_utils.coords_pixel_to_local((-1,51.2), dummy_data)
  assert (0.5, 1.0) == cv_utils.coords_pixel_to_local((128,512), dummy_data)

def test_coords_local_to_pixel():
  dummy_data = helper_dummychannels()
  assert (25, 51) == cv_utils.coords_local_to_pixel((0.1, 0.2), dummy_data)
  assert (25, 256) == cv_utils.coords_local_to_pixel((0.1, 1.0), dummy_data)
  assert (0, 256) == cv_utils.coords_local_to_pixel((-1.0, 2.0), dummy_data)

def test_distance_betweencoords():
  assert np.sqrt(8) == cv_utils.distance_betweencoords((1,1), (3,3))
  assert 2 == cv_utils.distance_betweencoords((0,0), (0,2))
  assert 2 == cv_utils.distance_betweencoords((0,0), (2,0))

def test_distance_to_origin():
  assert np.sqrt(18) == cv_utils.distance_to_origin((3,3))
  assert 2 == cv_utils.distance_to_origin((0,2))
  assert 2 == cv_utils.distance_to_origin((2,0))

def test_draw_rectangle():
  dummy_data = helper_dummychannels()

  result = cv_utils.draw_rectangle(dummy_data, (10,100), (128, 200), (255,127,0))
  expected = helper_getimage("dummy_data_rect_fill.jpg")
  assert np.all(np.equal(result, expected))

  result = cv_utils.draw_rectangle(dummy_data, (10,100), (128, 200), (255,127,0), -1)
  expected = helper_getimage("dummy_data_rect_fill.jpg")
  assert np.all(np.equal(result, expected))
  
  result = cv_utils.draw_rectangle(dummy_data, (10,100), (128, 200), (255,127,0), 10)
  expected = helper_getimage("dummy_data_rect_line.jpg")
  assert np.all(np.equal(result, expected))

  ## make sure we haven't been writing over dummy_data
  assert np.all(np.equal(dummy_data, helper_dummychannels()))

def test_draw_circle():
  dummy_data = helper_dummychannels()
  #Lets draw a circle with no fill
  result = cv_utils.draw_circle(dummy_data, (128,128), 64, (255,255,0), 10)
  
  # Lets draw a circle inside of it
  result = cv_utils.draw_circle(result, (128,128), 50, (255,127,0), -1)

  expected = helper_getimage("circle_and_arc.jpg")

  assert np.all(np.equal(expected, result))
  assert np.all(np.equal(helper_dummychannels(), dummy_data))

def test_draw_line():
  dummy_data = helper_dummychannels()
  result = cv_utils.draw_line(dummy_data, (0,0), (128,128), (255,220,0), 10)
  expected = helper_getimage("line.jpg")

  assert np.all(np.equal(expected, result))
  assert np.all(np.equal(helper_dummychannels(), dummy_data))


def test_draw_polygon():
  dummy_data = helper_dummychannels()
  pts = np.array([
      [50,50],
      [100, 50],
      [150, 150],
      [50, 150],
      ], np.int32)

  result = cv_utils.draw_polygon(dummy_data, pts, (255, 255, 0),5,True)
  pts = pts + 10
  result = cv_utils.draw_polygon(result, pts, (255, 127, 0),5,False)
  expected = helper_getimage("polygons_openclose.jpg")

  assert np.all(np.equal(result, expected))
  assert np.all(np.equal(dummy_data, helper_dummychannels()))
  assert result.shape == (256,256,3)

  # pts should be able to be just a list
  # result = cv_utils.draw_polygon(dummy_data, pts.tolist(), (255, 255, 0),5,True)
  # pts = pts + 10
  # result = cv_utils.draw_polygon(result, pts.tolist(), (255, 127, 0),5,False)
  # expected = helper_getimage("polygons_openclose.jpg")

  # assert np.all(np.equal(dummy_data, helper_dummychannels()))
  # assert result.shape == (256,256,3)

def test_translate_inpixels():
  pass

def test_translate_inlocal():
  pass

def test_crop_inpixels():
  pass

def test_crop_inlocal():
  pass

def test_rotate():
  pass

def test_flip():
  pass

def test_scale_inpixels():
  pass

def test_scale_inlocal():
  pass

def test_bit_and():
  pass

def test_bit_or():
  pass

def test_bit_xor():
  pass

def test_bit_not():
  pass

def test_apply_mask():
  pass

def test_blur():
  pass

def test_blur_kernel():
  pass

def test_blur_gausian():
  pass

def test_blur_median():
  pass

def test_blur_bilateral():
  pass

def test_denoise():
  pass

def test_sharpen():
  pass

def test_threshold_binary():
  pass

def test_threshold_binary_inverse():
  pass

def test_threshold_truncate():
  pass

def test_threshold_tozero():
  pass

def test_threshold_tozero_inverse():
  pass

def test_threshold_adaptive():
  pass

def test_threshold_otsu():
  pass

def test_morph_dilate():
  pass

def test_morph_erode():
  pass

def test_morph_open():
  pass

def test_morph_close():
  pass

def test_edge_sobel_x():
  pass

def test_edge_sobel_y():
  pass

def test_edge_laplacian():
  pass

def test_edge_canny():
  pass

def test_edge_canny_blur():
  pass

def test_perspective_transform():
  pass

def test_compute_contours():
  pass

def test_contours_sort_left_to_right():
  pass

def test_contours_sort_top_down():
  pass

def test_contours_sort_from_origin():
  pass

def test_contours_sort_by_area():
  pass

def test_contours_centroid():
  pass

def test_contours_highlight():
  pass

def test_contour_approximation():
  pass

def test_contour_hull_approximation():
  pass

def test_match_shape():
  pass

def test_match_hough_lines():
  pass

def test_match_prob_hough_lines():
  pass

def test_match_circle():
  pass

def test_match_blobs():
  pass

def test_describe_corners():
  pass

def test_describe_good2track():
  pass

def test_describe_FAST():
  pass

def test_describe_BRIEF():
  pass

def test_describe_ORG():
  pass

def test_describe_HOG():
  pass

def test_match_face():
  pass

def test_match_eyes():
  pass

def test_match_facial_landmarks():
  pass

def test_compute_faceswap():
  pass

def test_filter_by_color():
  pass




