from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import PIL.Image as Image
import random
import numpy as np
import cv2
import time


def get_frames_data(filename, num_frames_per_clip=16):
  ''' Given a directory containing extracted frames, return a video clip of
  (num_frames_per_clip) consecutive frames as a list of np arrays '''
  ret_arr = []
  s_index = 0
  #print(filename)
  for parent, dirnames, filenames in os.walk(filename):
    #print(len(filenames))
    if(len(filenames)<num_frames_per_clip):
      return [], s_index
    filenames = sorted(filenames)
    #s_index = random.randint(0, len(filenames) - num_frames_per_clip)
    while(s_index+num_frames_per_clip<=len(filenames)):
      arr = []
      for i in range(s_index, s_index + num_frames_per_clip):
        image_name = str(filename) + '/' + str(filenames[i])
        img = Image.open(image_name)
        img_data = np.array(img)
        arr.append(img_data)
      ret_arr.append(arr)
      s_index += 8
  return ret_arr, s_index


def read_clip_and_label(filename, batch_size, start_pos=-1, num_frames_per_clip=16, crop_size=112, shuffle=False):
  lines = open(filename,'r')
  read_dirnames = []
  data = []
  label = []
  batch_index = 0
  next_batch_start = -1
  lines = list(lines)

  #print(lines)
  np_mean = np.load('crop_mean.npy').reshape([num_frames_per_clip, crop_size, crop_size, 3])
  # Forcing shuffle, if start_pos is not specified
  if start_pos < 0:
    shuffle = True
  if shuffle:
    video_indices = range(len(lines))
    random.seed(time.time())
    random.shuffle(video_indices)
  else:
    # Process videos sequentially
    video_indices = range(start_pos, len(lines))
  for index in video_indices:
    if(batch_index>=batch_size):
      next_batch_start = index
      break
    line = lines[index].strip('\n').split()
    dirname = line[0]
    tmp_label = line[1]
    #print(dirname)
    if not shuffle:
      pass
      #print("Loading a video clip from {}...".format(dirname))
    tmp_data, _ = get_frames_data(dirname, num_frames_per_clip)
    #print(len(tmp_data))
    
    if(len(tmp_data)!=0):
      for i in xrange(len(tmp_data)):
        img_datas = [];
        for j in xrange(len(tmp_data[i])):
          img = Image.fromarray(tmp_data[i][j].astype(np.uint8))
          if(img.width>img.height):
            scale = float(crop_size)/float(img.height)
            img = np.array(cv2.resize(np.array(img),(int(img.width * scale + 1), crop_size))).astype(np.float32)
          else:
            scale = float(crop_size)/float(img.width)
            img = np.array(cv2.resize(np.array(img),(crop_size, int(img.height * scale + 1)))).astype(np.float32)
          crop_x = int((img.shape[0] - crop_size)/2)
          crop_y = int((img.shape[1] - crop_size)/2)
          #print(np_mean.shape)
          img = img[crop_x:crop_x+crop_size, crop_y:crop_y+crop_size,:] - np_mean[j]
          img_datas.append(img)
          #print(len(img_datas))
        data.append(img_datas)
        label.append(int(tmp_label))
        batch_index = batch_index + 1
        read_dirnames.append(dirname)

  # pad (duplicate) data/label if less than batch_size
  valid_len = len(data)
  pad_len = batch_size - valid_len
  #print(pad_len)
  if pad_len:
    for i in range(pad_len):
      data.append(img_datas)
      label.append(int(tmp_label))

  np_arr_data = np.array(data).astype(np.float32)
  np_arr_label = np.array(label).astype(np.int64)
  #print(np_arr_data.shape)
  #print(np_arr_label.shape)
  #print(next_batch_start)
  return np_arr_data, np_arr_label, next_batch_start, read_dirnames, valid_len



test_list_file = 'list/image.list'
read_clip_and_label(
                    test_list_file,
                    1,
                    start_pos=0
                    )
