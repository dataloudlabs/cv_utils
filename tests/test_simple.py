import cv_utils
import pickle
import numpy as np
import cv2

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
  return result.copy()

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
  dummy_data = helper_dummychannels()
  result = cv_utils.translate_inpixels(dummy_data, 20,20)
  expected = helper_getimage("translated_1.jpg")
  assert np.all(np.equal(result, expected))

  result = cv_utils.translate_inpixels(dummy_data, -20,-20)
  expected = helper_getimage("translated_2.jpg")
  assert np.all(np.equal(result, expected))

  assert np.all(np.equal(dummy_data, helper_dummychannels()))

def test_translate_inlocal():
  dummy_data = helper_dummychannels()
  result = cv_utils.translate_inlocal(dummy_data, 0.5,0.5)
  expected = helper_getimage("translated_3.jpg")
  assert np.all(np.equal(result, expected))

  result = cv_utils.translate_inlocal(dummy_data, -0.5,0.5)
  expected = helper_getimage("translated_4.jpg")
  assert np.all(np.equal(result, expected))

  assert np.all(np.equal(dummy_data, helper_dummychannels()))

def test_crop_inpixels():
  image_data = helper_getimage("cake.jpg")
  result = cv_utils.crop_inpixels(image_data, (400,80), (650,490))
  expected = helper_getimage("cake_crop.jpg")

  assert np.all(np.equal(result, expected))
  assert np.all(np.equal(image_data, helper_getimage("cake.jpg")))

def test_crop_inlocal():
  image_data = helper_getimage("cake.jpg")
  result = cv_utils.crop_inlocal(image_data, (0.5,0.5), (1.0,1.0))
  expected = helper_getimage("cake_crop2.jpg")

  assert np.all(np.equal(result, expected))
  assert np.all(np.equal(image_data, helper_getimage("cake.jpg")))

def test_rotate():
  dummy_data = helper_dummychannels()
  result = cv_utils.rotate(dummy_data, 90, pivot_coords=None, scale=1.0)
  expected = helper_getimage("rotate_90.jpg")
  assert np.all(np.equal(result, expected))

  result = cv_utils.rotate(dummy_data, -45, pivot_coords=None, scale=1.0)
  expected = helper_getimage("rotate_45.jpg")
  assert np.all(np.equal(result, expected))

  result = cv_utils.rotate(dummy_data, 45, pivot_coords=(0,0), scale=1.0)
  expected = helper_getimage("rotate_45_corner.jpg")
  assert np.all(np.equal(result, expected))

  result = cv_utils.rotate(dummy_data, 45, pivot_coords=None, scale=0.5)
  expected = helper_getimage("rotate_45_scale.jpg")
  assert np.all(np.equal(result, expected))

def test_flip():
  dummy_data = helper_dummychannels()
  result = cv_utils.flip(dummy_data, 0)
  expected = helper_getimage("flip_horiz.jpg")
  assert np.all(np.equal(result, expected))

  result = cv_utils.flip(dummy_data, 1)
  expected = helper_getimage("flip_vert.jpg")
  assert np.all(np.equal(result, expected))

  assert np.all(np.equal(dummy_data, helper_dummychannels()))

def test_scale_inpixels():
  dummy_data = helper_dummychannels()
  result = cv_utils.scale_inpixels(dummy_data, 300, 300)
  expected = helper_getimage("scale2.jpg")
  assert np.all(np.equal(result, expected))
  assert np.all(np.equal(dummy_data, helper_dummychannels()))

def test_scale_inlocal():
  dummy_data = helper_dummychannels()
  result = cv_utils.scale_inlocal(dummy_data, 0.5, 0.75)
  expected = helper_getimage("scale1.jpg")
  assert np.all(np.equal(result, expected))
  assert np.all(np.equal(dummy_data, helper_dummychannels()))

def test_bit_and():
  mask1 = helper_getimage("mask1.jpg")
  mask2 = helper_getimage("mask2.jpg")
  expected = helper_getimage("and.jpg")
  result = cv_utils.bit_and(mask1, mask2)

  assert np.all(np.equal(result, expected))

def test_bit_or():
  mask1 = helper_getimage("mask1.jpg")
  mask2 = helper_getimage("mask2.jpg")
  expected = helper_getimage("or.jpg")
  result = cv_utils.bit_or(mask1, mask2)

  assert np.all(np.equal(result, expected))

def test_bit_xor():
  mask1 = helper_getimage("mask1.jpg")
  mask2 = helper_getimage("mask2.jpg")
  expected = helper_getimage("xor.jpg")
  result = cv_utils.bit_xor(mask1, mask2)

  assert np.all(np.equal(result, expected))

def test_bit_not():
  mask1 = helper_getimage("mask1.jpg")
  expected = helper_getimage("not.jpg")
  result = cv_utils.bit_not(mask1)

  assert np.all(np.equal(result, expected))

def test_apply_mask():
  dummy_data = helper_dummychannels()
  mask = helper_getimage("and.jpg")
  expected = helper_getimage("and_apply.jpg")
  result = cv_utils.apply_mask(dummy_data, mask)

  assert np.all(np.equal(result, expected))
  assert np.all(np.equal(dummy_data, helper_dummychannels()))

def test_blur_box():
  image_data = helper_getimage("cake.jpg")
  
  expected = helper_getimage("blur_box_100.jpg")
  result = cv_utils.blur_box(image_data, 100)
  assert np.all(np.equal(result, expected))
  
  expected = helper_getimage("blur_box_7.jpg")
  result = cv_utils.blur_box(image_data, 7)
  assert np.all(np.equal(result, expected))

  assert np.all(np.equal(image_data, helper_getimage("cake.jpg")))

def test_blur_gausian():
  image_data = helper_getimage("cake.jpg")
  result = cv_utils.blur_gausian(image_data, 101)
  expected = helper_getimage("blur_gau_101.jpg")
  assert np.all(np.equal(result, expected))

  result = cv_utils.blur_gausian(image_data, 7)
  expected = helper_getimage("blur_gau_7.jpg")
  assert np.all(np.equal(result, expected))

  assert np.all(np.equal(image_data, helper_getimage("cake.jpg")))

def test_blur_median():
  image_data = helper_getimage("cake.jpg")
  result = cv_utils.blur_median(image_data, 101)
  expected = helper_getimage("blur_median_101.jpg")
  assert np.all(np.equal(result, expected))

  result = cv_utils.blur_median(image_data, 7)
  expected = helper_getimage("blur_median_7.jpg")
  assert np.all(np.equal(result, expected))

  assert np.all(np.equal(image_data, helper_getimage("cake.jpg")))

def test_blur_bilateral():
  image_data = helper_getimage("cake.jpg")
  expected = helper_getimage("blur_bilat.jpg")

  result = cv_utils.blur_bilateral(image_data)
  assert np.all(np.equal(result, expected))

  result = cv_utils.blur_bilateral(image_data, 9, 75, 75)
  assert np.all(np.equal(result, expected))

  assert np.all(np.equal(image_data, helper_getimage("cake.jpg")))

def test_denoise():
  image_data = helper_getimage("cake.jpg")
  result = cv_utils.denoise(image_data, 1)
  expected = helper_getimage("denoise1.jpg")
  assert np.all(np.equal(result, expected))

  result = cv_utils.denoise(image_data, 33)
  expected = helper_getimage("denoise2.jpg")
  assert np.all(np.equal(result, expected))
  assert np.all(np.equal(image_data, helper_getimage("cake.jpg")))

def test_sharpen():
  image_data = helper_getimage("cake.jpg")
  result = cv_utils.sharpen(image_data)
  expected=helper_getimage("sharpen.jpg")

  assert np.all(np.equal(result, expected))
  assert np.all(np.equal(image_data, helper_getimage("cake.jpg")))

def test_threshold_binary():
  dummy_data = helper_dummychannels()
  result = cv_utils.threshold_binary(dummy_data, 80, 255)
  expected = helper_getimage("tresh_binary255.jpg")
  assert np.all(np.equal(result, expected))
  
  result = cv_utils.threshold_binary(dummy_data, 80, 100)
  expected = helper_getimage("tresh_binary100.jpg")
  assert np.all(np.equal(result, expected))

  assert np.all(np.equal(dummy_data, helper_dummychannels()))

def test_threshold_binary_inverse():
  dummy_data = helper_dummychannels()

  result = cv_utils.threshold_binary_inverse(dummy_data, 80, 255)
  expected = helper_getimage("tresh_binaryi255.jpg")
  assert np.all(np.equal(result, expected))

  result = cv_utils.threshold_binary_inverse(dummy_data, 80, 100)
  expected = helper_getimage("tresh_binaryi100.jpg")
  assert np.all(np.equal(result, expected))

  assert np.all(np.equal(dummy_data, helper_dummychannels()))

def test_threshold_truncate():
  dummy_data = helper_dummychannels()
  result = cv_utils.threshold_truncate(dummy_data, 200, 255)
  expected = helper_getimage("tresh_trunc200.jpg")
  assert np.all(np.equal(result, expected))

  result = cv_utils.threshold_truncate(dummy_data, 140, 255)
  expected = helper_getimage("tresh_trunc140.jpg")
  assert np.all(np.equal(result, expected))

  assert np.all(np.equal(dummy_data, helper_dummychannels()))

def test_threshold_tozero():
  dummy_data = helper_dummychannels()

  result = cv_utils.threshold_tozero(dummy_data, 127, 255)
  expected = helper_getimage("tresh_tozero.jpg")
  assert np.all(np.equal(result, expected))

  assert np.all(np.equal(dummy_data, helper_dummychannels()))

def test_threshold_tozero_inverse():
  dummy_data = helper_dummychannels()

  result = cv_utils.threshold_tozero_inverse(dummy_data, 127, 255)
  expected = helper_getimage("tresh_tozeroi.jpg")
  assert np.all(np.equal(result, expected))

  assert np.all(np.equal(dummy_data, helper_dummychannels()))

def test_threshold_adaptive():
  image_data = helper_getimage("cake.jpg")
  image_data = cv_utils.convert_to_gray(image_data)

  result = cv_utils.threshold_adaptive(image_data)
  expected = helper_getimage("tresh_ada.jpg")
  assert np.all(np.equal(result, expected))

  result = cv_utils.threshold_adaptive(image_data, blocksize=7)
  expected = helper_getimage("tresh_ada7.jpg")
  assert np.all(np.equal(result, expected))

  result = cv_utils.threshold_adaptive(image_data, blocksize=15)
  expected = helper_getimage("tresh_ada15.jpg")
  assert np.all(np.equal(result, expected))

  assert np.all(np.equal(image_data, cv_utils.convert_to_gray(helper_getimage("cake.jpg"))))

def test_threshold_otsu():
  image_data = helper_getimage("cake.jpg")
  image_data = cv_utils.convert_to_gray(image_data)

  result = cv_utils.threshold_otsu(image_data, 240)
  expected = helper_getimage("tresh_otsu.jpg")
  assert np.all(np.equal(result, expected))
  assert np.all(np.equal(image_data, cv_utils.convert_to_gray(helper_getimage("cake.jpg"))))

def test_threshold_gaussianotsu():
  image_data = helper_getimage("cake.jpg")
  image_data = cv_utils.convert_to_gray(image_data)

  result = cv_utils.threshold_gaussianotsu(image_data, 240)
  expected = helper_getimage("tresh_gotsu.jpg")
  assert np.all(np.equal(result, expected))
  assert np.all(np.equal(image_data, cv_utils.convert_to_gray(helper_getimage("cake.jpg"))))

def test_morph_dilate():
  image_data = helper_getimage("hello.png")
  _, image_data = cv2.threshold(image_data, 0, 255, cv2.THRESH_BINARY_INV)

  result = cv_utils.morph_dilate(image_data, 5)
  expected = helper_getimage("dilate_5.jpg")
  assert np.all(np.equal(result, expected))
  
  result = cv_utils.morph_dilate(image_data, (5,5))
  expected = helper_getimage("dilate_5x5.jpg")
  assert np.all(np.equal(result, expected))

def test_morph_erode():
  image_data = helper_getimage("hello.png")
  _, image_data = cv2.threshold(image_data, 0, 255, cv2.THRESH_BINARY_INV)

  result = cv_utils.morph_erode(image_data, 5)
  expected = helper_getimage("erode_5.jpg")
  assert np.all(np.equal(result, expected))
  
  result = cv_utils.morph_erode(image_data, (5,5))
  expected = helper_getimage("erode_5x5.jpg")
  assert np.all(np.equal(result, expected))

def test_morph_open():
  image_data = helper_getimage("hello.png")
  _, image_data = cv2.threshold(image_data, 0, 255, cv2.THRESH_BINARY_INV)

  result = cv_utils.morph_open(image_data, 5)
  expected = helper_getimage("open_5.jpg")
  assert np.all(np.equal(result, expected))
  
  result = cv_utils.morph_open(image_data, (5,5))
  expected = helper_getimage("open_5x5.jpg")
  assert np.all(np.equal(result, expected))

def test_morph_close():
  image_data = helper_getimage("hello.png")
  _, image_data = cv2.threshold(image_data, 0, 255, cv2.THRESH_BINARY_INV)

  result = cv_utils.morph_close(image_data, 5)
  expected = helper_getimage("close_5.jpg")
  assert np.all(np.equal(result, expected))
  result = cv_utils.morph_close(image_data, (5,5))
  expected = helper_getimage("close_5x5.jpg")
  assert np.all(np.equal(result, expected))

def test_edge_sobel_x():
  image_data = helper_getimage("car.jpg")
  image_data = cv_utils.convert_to_gray(image_data)

  result = cv_utils.edge_sobel_x(image_data, 5)
  expected = helper_getimage("edge_sobelx.jpg")
  assert np.all(np.equal(result, expected))

def test_edge_sobel_y():
  image_data = helper_getimage("car.jpg")
  image_data = cv_utils.convert_to_gray(image_data)

  result = cv_utils.edge_sobel_y(image_data, 5)
  expected = helper_getimage("edge_sobely.jpg")
  assert np.all(np.equal(result, expected))

def test_edge_laplacian():
  image_data = helper_getimage("car.jpg")
  image_data = cv_utils.convert_to_gray(image_data)

  result = cv_utils.edge_laplacian(image_data)
  expected = helper_getimage("edge_laplacian.jpg")
  assert np.all(np.equal(result, expected))

def test_edge_canny():
  image_data = helper_getimage("car.jpg")
  image_data = cv_utils.convert_to_gray(image_data)

  result = cv_utils.edge_canny(image_data, 50, 120)
  expected = helper_getimage("edge_canny.jpg")
  assert np.all(np.equal(result, expected))

def test_edge_canny_blur():
  image_data = helper_getimage("car.jpg")
  image_data = cv_utils.convert_to_gray(image_data)

  result = cv_utils.edge_canny_blur(image_data, 50, 120)
  expected = helper_getimage("edge_canny_blur.jpg")
  assert np.all(np.equal(result, expected))

def test_perspective_affine():
  image_data = helper_getimage("affine.jpg")
  
  points_A = np.float32([[320,15], [700,215], [85,610]])
  points_B = np.float32([[0,0], [420,0], [0,594]])

  result = cv_utils.perspective_affine(image_data, points_A, points_B)
  expected = helper_getimage("affine2.jpg")

  assert np.all(np.equal(result, expected))
  assert np.all(np.equal(image_data, helper_getimage("affine.jpg")))

def test_perspective_nonaffine():
  image_data = helper_getimage("keyboard.jpg")
  
  points_A = np.float32([[255,65], [935,330], [50,220], [730,700]])
  points_B = np.float32([[0,0], [700,0], [0,300], [700,300]])
  newimage_image_size = (700, 300)

  result = cv_utils.perspective_nonaffine(image_data, points_A, points_B, newimage_image_size)
  expected = helper_getimage("nonaffine.jpg")
  assert np.all(np.equal(result, expected))

  result = cv_utils.perspective_nonaffine(image_data, points_A, points_B)
  expected = helper_getimage("nonaffine2.jpg")
  assert np.all(np.equal(result, expected))

  assert np.all(np.equal(image_data, helper_getimage("keyboard.jpg")))

def test_compute_contours():
  image_data = helper_getimage("color_shapes.jpg")
  expected = helper_getimage("color_shapes_2.jpg")

  contours, hierarchy = cv_utils.compute_contours(image_data)
  assert len(contours) == 0

  contours, hierarchy = cv_utils.compute_contours(image_data, 254, 255)
  cv2.drawContours(image_data, contours, -1, (0,0,255), 10)
  assert np.all(np.equal(image_data, expected))

def test_contour_centroid():
  image_data = helper_getimage("color_shapes.jpg")
  expected = [(569, 199), (359, 199), (779, 199), (149, 199)]

  contours, hierarchy = cv_utils.compute_contours(image_data, 254, 255)
  result = map(cv_utils.contour_centroid, contours)
  assert np.all(np.equal(result, expected))

def test_contour_area():
  image_data = helper_getimage("color_shapes.jpg")
  expected = [19878.0, 31460.0, 40398.0, 40398.0]

  contours, hierarchy = cv_utils.compute_contours(image_data, 254, 255)
  result = map(cv_utils.contour_area, contours)

  #print(expected)
  #print(result)

  assert np.all(np.equal(result, expected))

def test_contours_highlight():
  image_data = helper_getimage("color_shapes.jpg")
  expected = helper_getimage("color_shapes_2.jpg")

  contours, hierarchy = cv_utils.compute_contours(image_data, 254, 255)
  result = cv_utils.contours_highlight(image_data, contours, thickness=10)
  assert np.all(np.equal(result, expected))

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




