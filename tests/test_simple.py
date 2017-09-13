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

def test_contour_boundingbox():
  image_data = helper_getimage("house.png")
  contours, hier = cv_utils.compute_contours(image_data, 127,255)
  
  expected = [((148, 254), (197, 335)),
              ((107, 238), (214, 344)),
              ((97, 150), (263, 344)),
              ((79, 136), (80, 138)),
              ((75, 135), (78, 138)),
              ((79, 29), (251, 136)),
              ((64, 10), (278, 363))]

  result = map(cv_utils.contour_boundingbox, contours)
  
  assert np.all(np.equal(expected, result))

def test_contours_highlight():
  image_data = helper_getimage("color_shapes.jpg")
  expected = helper_getimage("color_shapes_2.jpg")

  contours, hierarchy = cv_utils.compute_contours(image_data, 254, 255)
  result = cv_utils.contours_highlight(image_data, contours, thickness=10)
  assert np.all(np.equal(result, expected))

def test_contour_approximation():
  image_data = helper_getimage("house.png")
  expected = helper_getimage("house3.png")

  contours, hier = cv_utils.compute_contours(image_data, 127,255)

  approxs = [ cv_utils.contour_approximation(x,0.001) for x in contours ]
  result = cv_utils.contours_highlight(image_data, approxs)
  
  assert np.all(np.equal(expected, result))

def test_contour_hull_approximation():
  data = helper_getimage("flower3.png")
  expected = helper_getimage("flower4.png")

  contours, hier = cv_utils.compute_contours(data, 176, 255)
  hulls = map(cv_utils.contour_hull_approximation, contours)
  result = cv_utils.contours_highlight(data, hulls)

  assert np.all(np.equal(result, expected))

def test_match_shape():
  image_data = helper_getimage("shapes3")
  template_data = helper_getimage("shapes4")
  template_c, template_h = cv_utils.compute_contours(template_data, 0, 255)
  search_c, search_h = cv_utils.compute_contours(image_data, 0, 255)

  result = [ cv_utils.match_shape(x, template_c[0], cv2.CONTOURS_MATCH_I3) for x in search_c]
  expected = [0.18902099881889753, 6.19450632585382, 0.1594613138288294]

  #https://stackoverflow.com/questions/46176571/interpreting-opencv-cv2-matchshapes-weird-values-using-python
  assert np.all(np.equal(result, expected))

def test_match_hough_lines():
  image_data = helper_getimage("skyscrapper.jpg")
  image_data = cv_utils.convert_to_gray(image_data)
  image_data = cv_utils.edge_canny(image_data, 50, 150)
  
  lines = cv_utils.match_hough_lines(image_data, 223)
  result = helper_getimage("skyscrapper.jpg")
  map(lambda x: cv2.line(result,x[0],x[1],(0,0,255),2), lines)

  expected = helper_getimage("skyscrapper3.jpg")
  assert np.all(np.equal(result, expected))

  result = cv_utils.match_hough_lines(image_data, 450)
  expected = [[(598, -1032), (703, 964)],
              [(595, -1032), (700, 964)],
              [(-1011, 668), (987, 703)],
              [(407, -1212), (1091, 666)]]

  assert np.all(np.equal(result, expected))

def test_match_prob_hough_lines():
  image_data = helper_getimage("skyscrapper2.jpg")
  image_data = cv_utils.convert_to_gray(image_data)
  image_data = cv_utils.edge_canny(image_data, 100, 170)

  lines = cv_utils.match_prob_hough_lines(image_data, 220, 10, 5)
  result = helper_getimage("skyscrapper2.jpg")
  map(lambda x: cv2.line(result,x[0],x[1],(0,0,255),2), lines)

  expected = helper_getimage("skyscrapper4.jpg")
  assert np.all(np.equal(result, expected))

  result = cv_utils.match_hough_lines(image_data, 700, 100, 5)

  # expected = [[(1111, 422), (1251, 331)],
  #            [(873, 773), (992, 696)],
  #            [(969, 510), (1081, 437)],
  #            [(962, 524), (1089, 441)]]

  expected = [[(1450, 1000), (1450, -1000)], 
              [(1150, 1000), (1150, -1000)], 
              [(950, 1000), (950, -1000)], 
              [(750, 1000), (750, -1000)]]

  assert np.all(np.equal(result, expected))

def test_match_circle():
  data = helper_getimage("circles.jpg")
  data = cv_utils.crop_inlocal(data, [0,0], [0.37 ,0.37])
  data = cv_utils.convert_to_gray(data)
  data = cv_utils.threshold_binary(data, 10, 255)

  centers, radius = cv_utils.match_circle(data, 1)
  expected = [((199, 199), 25), 
              ((199, 40), 25), 
              ((40, 40), 24), 
              ((119, 40), 25)]

  result = [ x for x in zip(centers, radius) ]

  assert np.all(np.equal(result, expected))

def test_match_blobs():
  image_data = helper_getimage("cereals2.jpg")
  centers, radius, k = cv_utils.match_blobs(image_data)
  centers = [map(int, x) for x in centers]
  radius = map(int, radius)
  
  expected = [([883, 442], 25),
              ([340, 279], 21),
              ([781, 436], 25),
              ([175, 385], 26),
              ([67, 301], 24)]

  result = [ x for x in zip(centers, radius) ]

  assert np.all(np.equal(result, expected))

def test_describe_corners():
  image_data = helper_getimage('car.jpg')
  image_data = cv_utils.crop_inlocal(image_data, (0.5, 0.1), (1.0,0.6))
  result = image_data.copy()
  expected = helper_getimage("harris_corners.jpg")

  harris_corners = cv_utils.describe_corners(image_data, 3 ,3 ,0.05)
  harris_corners = cv2.dilate(harris_corners, None)
  result[harris_corners>0.025*harris_corners.max()]=[255,127,0]
  
  assert harris_corners.shape == image_data.shape[:2]
  assert np.all(np.equal(result, expected))

def test_describe_good2track():
  # Load image then grayscale
  image_data = helper_getimage('car.jpg')
  result = cv_utils.describe_good2track(image_data, 300)
  expected = [[ 499.,  324.],
              [ 225.,  532.],
              [  66.,  179.],
              [ 791.,  566.],
              [ 784.,  189.],
              [ 403.,   19.]]

  assert np.all(np.equal(result, expected))

def test_describe_FAST():
  # Load image then grayscale
  image_data = helper_getimage('car.jpg')
  image_data = cv_utils.crop_inlocal(image_data, (0.65, 0.0), (1.0,0.15))

  centers, radius, keypoints = cv_utils.describe_FAST(image_data)

  expected = [(111.0, 3.0), (189.0, 3.0), (119.0, 7.0), (95.0, 8.0),
             (122.0, 10.0), (171.0, 10.0), (222.0, 10.0), (92.0, 14.0),
             (158.0, 14.0), (155.0, 15.0), (138.0, 16.0), (141.0, 16.0),
             (148.0, 16.0), (144.0, 17.0), (86.0, 18.0), (82.0, 32.0),
             (80.0, 36.0), (57.0, 37.0), (59.0, 38.0), (64.0, 39.0), 
             (53.0, 40.0), (72.0, 40.0), (50.0, 43.0), (35.0, 45.0), 
             (40.0, 45.0), (45.0, 47.0), (50.0, 47.0), (34.0, 48.0), 
             (36.0, 48.0), (41.0, 48.0), (22.0, 49.0), (15.0, 52.0), 
             (16.0, 54.0), (78.0, 63.0)]

  assert np.all(np.equal(centers, expected))

def test_describe_BRIEF():
  # Load image then grayscale
  image_data = helper_getimage('car.jpg')
  image_data = cv_utils.crop_inlocal(image_data, (0.7, 0.1), (1.0,0.6))

  centers, radius, keypoints = cv_utils.describe_BRIEF(image_data)

  expected_centers = [(114.0, 71.0), (72.0, 73.0), (85.0, 75.0), (90.0, 77.0),
                      (98.0, 77.0), (71.0, 79.0), (93.0, 78.0), (160.0, 78.0),
                      (96.0, 86.0), (172.0, 106.0), (69.0, 110.0), (175.0, 111.0),
                      (167.0, 118.0), (172.0, 118.0), (72.0, 124.0), (165.0, 123.0),
                      (164.0, 126.0), (173.0, 127.0), (71.0, 135.0), (106.0, 193.0),
                      (175.0, 224.0), (159.0, 236.0)]

  expected_radius = [6.0, 6.0, 8.0, 6.0, 6.0,
                     4.0, 4.0, 32.0, 8.0, 12.0,
                     6.0, 22.0, 12.0, 4.0, 4.0, 16.0,
                     22.0, 6.0, 8.0, 4.0, 6.0, 22.0]

  assert np.all(np.equal(expected_centers, centers))
  assert np.all(np.equal(expected_radius, radius))

def test_describe_ORB():
  # Load image then grayscale
  image_data = helper_getimage('car.jpg')
  image_data = cv_utils.crop_inlocal(image_data, (0.7, 0.24), (0.9,0.35))
  
  c, r, k = cv_utils.describe_ORB(image_data)

  expected = [(54.0, 31.0), (103.0, 32.0), (106.0, 32.0), (112.0, 32.0), 
              (57.0, 33.0), (117.0, 33.0), (59.0, 34.0), (95.0, 34.0), 
              (121.0, 34.0), (123.0, 34.0), (74.0, 35.0), (103.0, 35.0), 
              (126.0, 35.0), (62.0, 36.0), (68.0, 36.0), (70.0, 36.0), 
              (72.0, 36.0), (101.0, 36.0)]

  assert np.all(np.equal(c, expected))

def test_describe_HOG():
  image_data = helper_getimage("dogs.jpg")

  gradients, hog_feats = cv_utils.describe_HOG(image_data)
  assert gradients.shape == (53, 80, 9)

  gradients, hog_feats = cv_utils.describe_HOG(image_data, cellsize=(9,9), nbins=11)
  assert gradients.shape == (47, 71, 11) # (image_h/cell_zise, image_w//cell_size, angular nbins)


def test_match_face():
  image_data = helper_getimage("Patrick_Stewart_and_Hugh_Jackman_Press_Conference_Logan_Berlinale_2017_01.jpg")
  result = cv_utils.match_face(image_data)
  expected = [[(489, 124), (688, 323)], [(237, 260), (427, 450)]]

  assert np.all(np.equal(result, expected))

def test_match_eyes():
  image_data = helper_getimage("Patrick_Stewart_and_Hugh_Jackman_Press_Conference_Logan_Berlinale_2017_01.jpg")
  result = cv_utils.match_eyes(image_data)
  expected = [[(529, 189), (568, 228)], 
              [(589, 174), (641, 226)], 
              [(268, 307), (318, 357)]]

  assert np.all(np.equal(result, expected))

def test_match_facial_landmarks():
  pass

def test_compute_faceswap():
  pass

def test_filter_by_color():
  pass




