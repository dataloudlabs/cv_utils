import requests
import bz2

def haar_frontfacedefault(target_path):
  print "Downloading HAARCASCADE_FRONTALFACE_DEFAULT"
  url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
  r = requests.get(url)
  with open(target_path, "wb") as f:
      f.write(r.content)

  return target_path

def haar_eye(target_path):
  print "Downloading HAARCASCADE_EYE"
  url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml"
  r = requests.get(url)
  with open(target_path, "wb") as f:
      f.write(r.content)

  return target_path

def dlib_facelandmarks(target_path):
  print "Downloading Face Landmarks"
  url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
  r = requests.get(url)
  with open(target_path, 'wb') as new_file:
    new_file.write(bz2.decompress(r.content))

  return target_path