
import cv2
import tensorflow as tf
import argparse
import numpy as np
import multiprocessing
import os
from os import listdir
from os.path import isfile, join
from random import shuffle
import sys
import xml.etree.ElementTree

from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from tensorpack.tfutils.varreplace import remap_variables
from tensorpack.tfutils.varreplace import freeze_variables
from tensorpack.dataflow import dataset
from tensorpack.utils.gpu import get_nr_gpu

from dorefa import get_dorefa
from evaluate import *

DEMO_DATASET = 0

if DEMO_DATASET == 0:
	IMAGE_WIDTH = 640.0
	IMAGE_HEIGHT = 360.0
else:
	IMAGE_WIDTH = 640.0
	IMAGE_HEIGHT = 480.0


BITW = 1
BITA = 5
BITG = 32
BATCH_SIZE = 128

MONITOR = 1
REAL_IMAGE = 0
DEBUG_DATA_PIPELINE = 0


NUM_SQUEEZE_FILTERS = 32
NUM_EXPAND_FILTERS = 96

square_size = 224
height_width = 14
down_sample_factor = square_size/height_width


classes = [
	['boat',0],
	['building',1],
	['car',2],
	['drone',3],
	['group',4],
	['horseride',5],
	['paraglider',6],
	['person',7],
	['riding',8],
	['truck',9],
	['wakeboard',10],
	['whale',11]
]

def line_intersection_union(line1_min, line1_max, line2_min, line2_max):
	intersection = 0
	intersect_state = 0
	if line1_min <= line2_min and line2_min < line1_max:
		if line2_max > line1_max:
			intersection = line1_max - line2_min
			intersect_state = 1
		else:
			intersection = line2_max - line2_min
			intersect_state = 2
	elif line2_min <= line1_min and line1_min < line2_max:
		if line1_max > line2_max:
			intersection = line2_max - line1_min
			intersect_state = 3
		else:
			intersection = line1_max - line1_min
			intersect_state = 4

	union = 0
	if intersection > 0:
		if intersect_state == 1:
			union = line2_max - line1_min
		elif intersect_state == 2:
			union = line1_max - line1_min
		elif intersect_state == 3:
			union = line1_max - line2_min
		else:
			union = line2_max - line2_min

	return intersection, union

def intersection(rect1, rect2):
	
	x_intersection, x_union = line_intersection_union(rect1[0], rect1[1], rect2[0], rect2[1]) 
	y_intersection, y_union = line_intersection_union(rect1[2], rect1[3], rect2[2], rect2[3])

	intersection = x_intersection*y_intersection
	union = x_union*y_union

	if intersection > 0:
		scaled = float(intersection)/(height_width*height_width)
		if scaled > 0.1:
			return scaled
		else:
			return 0
	else:
		return 0

class DAC_Dataset(RNGDataFlow):
	def __init__(self, dataset_dir, train, all_classes):
		self.images = []
		
		if all_classes == 1:
			for directory in listdir(dataset_dir):
				for file in listdir(dataset_dir + '/' + directory):
					if '.jpg' in file:
						for c in classes:
							if c[0] in directory:
								label = c[1]
								break
						self.images.append([dataset_dir + '/' + directory + '/' + file, label])
		else:
			for file in listdir(dataset_dir):
				if '.jpg' in file:
					self.images.append([dataset_dir + '/' + file, 0])

		shuffle(self.images)
		if train == 0:
			self.images = self.images[0:1000]

	def get_data(self):
		for image in self.images:
			xml_name = image[0].replace('jpg','xml')

			im = cv2.imread(image[0], cv2.IMREAD_COLOR)
			im = cv2.resize(im, (square_size, square_size))
			im = im.reshape((square_size, square_size, 3))

			meta = None
			if os.path.isfile(image[0].replace('jpg','xml')):
				meta = xml.etree.ElementTree.parse(xml_name).getroot()

			label = np.array(image[1])

			bndbox = np.empty(shape = [0,4])    
			i = 0
			if meta is not None:
				for obj in meta.findall('object'):
					box = obj.find('bndbox')
					if box is not None:
						bndbox = np.append(bndbox,[[int(box.find('xmin').text),int(box.find('xmax').text),int(box.find('ymin').text),int(box.find('ymax').text)]],axis=0)
						bndbox[i,0] = int(bndbox[i,0]*(square_size/IMAGE_WIDTH))
						bndbox[i,1] = int(bndbox[i,1]*(square_size/IMAGE_WIDTH))
						bndbox[i,2] = int(bndbox[i,2]*(square_size/IMAGE_HEIGHT))
						bndbox[i,3] = int(bndbox[i,3]*(square_size/IMAGE_HEIGHT))
						i = i + 1
			count = i 
			iou = np.zeros( (height_width, height_width) )
			iou_count = np.zeros((height_width, height_width,count))
			for h in range(0, height_width):
				for w in range(0, height_width):
					rect = np.zeros(4)
					rect[0] = int(w*down_sample_factor)
					rect[1] = int((w+1)*down_sample_factor)
					rect[2] = int(h*down_sample_factor)
					rect[3] = int((h+1)*down_sample_factor)
					for i in range(count):
						if DEMO_DATASET == 0:
							if intersection(rect, bndbox[i,:])== 0.0:
								iou_count[h,w,i] = 0.0 
							else:
								iou_count[h,w,i] = 1.0
						else:
							if intersection(rect, bndbox[i,:]) < 0.5:
								iou_count[h,w,i] = 0.0
							else:
								iou_count[h,w,i] = 1.0
					for i in range(count):
						if DEMO_DATASET == 0:
							if intersection(rect, bndbox[i,:])== 0.0:
								iou[h,w]  = 0.0
							else:
								iou[h,w]  = 1.0
						else:
							if intersection(rect, bndbox[i,:]) < 0.5:
								iou[h,w]  = 0.0
							else:
								iou[h,w]  = 1.0
						if iou[h,w] > 0.0:
							break
					if DEBUG_DATA_PIPELINE == 1:
						if iou[h,w] > 0:
							cv2.rectangle(im, (int(rect[0]),int(rect[1])), (int(rect[2]),int(rect[3])), (0,0,iou[h,w]*255), 1)

			iou = iou.reshape( (height_width, height_width, 1) )

			valid = np.zeros((height_width, height_width, 4), dtype='float32')
			relative_bndboxes = np.zeros((height_width, height_width, 4), dtype='float32')
			for h in range(0, height_width):
				for w in range(0, height_width):
					if iou[h, w] > 0.0:
						valid[h,w,0] = 1.0
						valid[h,w,1] = 1.0
						valid[h,w,2] = 1.0
						valid[h,w,3] = 1.0
					for i in range(count):
						if iou_count[h,w,i] > 0.0:
							relative_bndboxes[h, w, 0] = bndbox[i,0] - w*down_sample_factor
							relative_bndboxes[h, w, 1] = bndbox[i,2] - h*down_sample_factor
							relative_bndboxes[h, w, 2] = bndbox[i,1] - w*down_sample_factor
							relative_bndboxes[h, w, 3] = bndbox[i,3] - h*down_sample_factor
							break
						else:
							relative_bndboxes[h, w] = np.zeros(4)

			if DEBUG_DATA_PIPELINE == 1:
				#cv2.rectangle(im, (bndbox['xmin'],bndbox['ymin']), (bndbox['xmax'],bndbox['ymax']), (255,0,0), 1)
				#cv2.imshow('image', im)
				cv2.waitKey(1000)

			yield [im, label, iou, valid, relative_bndboxes]

	def size(self):
		return len(self.images)

class Model(ModelDesc):
	def _get_inputs(self):
		return [InputDesc(tf.float32, [None, square_size, square_size, 3], 'input'),
				InputDesc(tf.int32, [None], 'label'),
				InputDesc(tf.float32, [None, height_width, height_width, 1], 'ious'),
				InputDesc(tf.float32, [None, height_width, height_width, 4], 'valids'),
				InputDesc(tf.float32, [None, height_width, height_width, 4], 'bndboxes')]

	def _build_graph(self, inputs):
		image, label, ious, valids, bndboxes = inputs
		image = tf.round(image)

		fw, fa, fg = get_dorefa(BITW, BITA, BITG)

		old_get_variable = tf.get_variable

		def monitor(x, name):
			if MONITOR == 1:
				return tf.Print(x, [x], message='\n\n' + name + ': ', summarize=1000, name=name)
			else:
				return x

		def new_get_variable(v):
			name = v.op.name
			if not name.endswith('W') or 'conv1' in name or 'conv_obj' in name or 'conv_box' in name:
				return v
			else:
				logger.info("Quantizing weight {}".format(v.op.name))
				if MONITOR == 1:
					return tf.Print(fw(v), [fw(v)], message='\n\n' + v.name + ', Quantized weights are:', summarize=100)
				else:
					return fw(v)

		def activate(x):
			if BITA == 32:
				return tf.nn.relu(x)
			else:
				return fa(tf.nn.relu(x))

		def bn_activate(name, x):
			x = BatchNorm(name, x)
			x = monitor(x, name + '_noact_out')
			return activate(x)

		def halffire(name, x, num_squeeze_filters, num_expand_3x3_filters, skip):
			out_squeeze = Conv2D('squeeze_conv_' + name, x, out_channel=num_squeeze_filters, kernel_shape=1, stride=1, padding='SAME')
			out_squeeze = bn_activate('bn_squeeze_' + name, out_squeeze)
			out_expand_3x3 = Conv2D('expand_3x3_conv_' + name, out_squeeze, out_channel=num_expand_3x3_filters, kernel_shape=3, stride=1, padding='SAME')
			out_expand_3x3 = bn_activate('bn_expand_3x3_' + name, out_expand_3x3)
			if skip == 0:
				return out_expand_3x3
			else:
				return tf.add(x, out_expand_3x3)

		def halffire_noact(name, x, num_squeeze_filters, num_expand_3x3_filters):
			out_squeeze = Conv2D('squeeze_conv_' + name, x, out_channel=num_squeeze_filters, kernel_shape=1, stride=1, padding='SAME')
			out_squeeze = bn_activate('bn_squeeze_' + name, out_squeeze)
			out_expand_3x3 = Conv2D('expand_3x3_conv_' + name, out_squeeze, out_channel=num_expand_3x3_filters, kernel_shape=3, stride=1, padding='SAME')
			return out_expand_3x3

		with 	remap_variables(new_get_variable), \
				argscope([Conv2D, FullyConnected], use_bias=False, nl=tf.identity), \
				argscope(BatchNorm, decay=0.9, epsilon=1e-4):

			image = monitor(image, 'image_out')

			l = Conv2D('conv1', image, out_channel=32, kernel_shape=3, stride=2, padding='SAME')
			l = bn_activate('bn1', l)
			l = monitor(l, 'conv1_out')

			l = MaxPooling('pool1', l, shape=3, stride=2, padding='SAME')
			l = monitor(l, 'pool1_out')

			l = halffire('fire1', l, NUM_SQUEEZE_FILTERS, NUM_EXPAND_FILTERS, 0)
			l = monitor(l, 'fire1_out')

			l = MaxPooling('pool2', l, shape=3, stride=2, padding='SAME')
			l = monitor(l, 'pool2_out')

			l = halffire('fire2', l, NUM_SQUEEZE_FILTERS, NUM_EXPAND_FILTERS, 0)
			l = monitor(l, 'fire2_out')

			l = MaxPooling('pool3', l, shape=3, stride=2, padding='SAME')
			l = monitor(l, 'pool3_out')

			l = halffire('fire3', l, NUM_SQUEEZE_FILTERS, NUM_EXPAND_FILTERS, 0)
			l = monitor(l, 'fire3_out')

			l = halffire('fire4', l, NUM_SQUEEZE_FILTERS, NUM_EXPAND_FILTERS, 0)
			l = monitor(l, 'fire4_out')

			l = halffire('fire5', l, NUM_SQUEEZE_FILTERS, NUM_EXPAND_FILTERS, 0)
			l = monitor(l, 'fire5_out')

			l = halffire('fire6', l, NUM_SQUEEZE_FILTERS, NUM_EXPAND_FILTERS, 0)
			l = monitor(l, 'fire6_out')

			l = halffire('fire7', l, NUM_SQUEEZE_FILTERS, NUM_EXPAND_FILTERS, 0)
			l = monitor(l, 'fire7_out')

			# Classification
			classify = Conv2D('conv_class', l, out_channel=12, kernel_shape=1, stride=1, padding='SAME')
			classify = bn_activate('bn_class', classify)
			classify = monitor(classify, 'conv_class_out')
			logits = GlobalAvgPooling('pool_class', classify)

			class_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
			class_loss = tf.reduce_mean(class_loss, name='cross_entropy_loss')

			wrong = prediction_incorrect(logits, label, 1, name='wrong-top1')
			add_moving_summary(tf.reduce_mean(wrong, name='train-error-top1'))

			# Object Detection
			l = tf.concat([l, classify], axis=3)

			objdetect = Conv2D('conv_obj', l, out_channel=1, kernel_shape=1, stride=1, padding='SAME')
			objdetect = tf.identity(objdetect, name='objdetect_out')
			objdetect_loss = tf.losses.hinge_loss(labels=ious, logits=objdetect)

			bndbox = Conv2D('conv_box', l, out_channel=4, kernel_shape=1, stride=1, padding='SAME')
			bndbox = tf.identity(bndbox, name='bndbox_out')
			bndbox = tf.multiply(bndbox, valids, name='mult0')
			bndbox_loss = tf.losses.mean_squared_error(labels=bndboxes, predictions=bndbox)

			# weight decay on all W of fc layers
			# reg_cost = regularize_cost('(fire7|conv_obj|conv_box).*/W', l2_regularizer(1e-5), name='regularize_cost')

			# cost = class_loss*objdetect_loss*bndbox_loss
			# cost = class_loss + objdetect_loss + bndbox_loss + reg_cost
			cost = class_loss + 10*objdetect_loss + bndbox_loss

			add_moving_summary(class_loss, objdetect_loss, bndbox_loss, cost)

		self.cost = cost

		tf.get_variable = old_get_variable

	def _get_optimizer(self):
		lr = tf.get_variable('learning_rate', initializer=1e-2, trainable=False)
		opt = tf.train.AdamOptimizer(lr, epsilon=1e-5)
		# lr = tf.get_variable('learning_rate', initializer=1e-1, trainable=False)
		# opt = tf.train.MomentumOptimizer(lr, momentum=0.9)
		return opt

def get_data(dataset_dir, train):
	if DEMO_DATASET == 0:
		all_classes = 1
	else:
		all_classes = 0
	ds = DAC_Dataset(dataset_dir, train, all_classes)
	ds = BatchData(ds, BATCH_SIZE, remainder=False)
	ds = PrefetchDataZMQ(ds, nr_proc=8, hwm=6)
	return ds

def get_config():
	logger.auto_set_dir()
	data_train = get_data(args.data, 1)
	data_test = get_data(args.data, 0)

	if DEMO_DATASET == 0:
		return TrainConfig(
			dataflow=data_train,
			callbacks=[
				ModelSaver(max_to_keep=10),
				HumanHyperParamSetter('learning_rate'),
				ScheduledHyperParamSetter('learning_rate', [(40, 0.001), (60, 0.0001), (90, 0.00001)])
				,InferenceRunner(data_test,
								[ScalarStats('cross_entropy_loss'),
								ClassificationError('wrong-top1', 'val-error-top1')])
				],
			model=Model(),
			max_epoch=150
		)
	else:
		return TrainConfig(
			dataflow=data_train,
			callbacks=[
				ModelSaver(max_to_keep=10),
				HumanHyperParamSetter('learning_rate'),
				ScheduledHyperParamSetter('learning_rate', [(100, 0.001), (200, 0.0001), (250, 0.00001)])
				],
			model=Model(),
			max_epoch=300
		)

def run_image(model, sess_init, image_dir):
	print('Running image!')

	output_names = ['objdetect_out', 'bndbox_out']

	pred_config = PredictConfig(
		model=model,
		session_init=sess_init,
		input_names=['input'],
		output_names=output_names
	)
	predictor = OfflinePredictor(pred_config)
	
	images = []
	metas = []
	for file in listdir(image_dir):
		if '.jpg' in file:
			images.append(file)
		if '.xml' in file:
			metas.append(file)

	images.sort()
	metas.sort()

	THRESHOLD = 0
	index = 0
	for image in images:
		meta = xml.etree.ElementTree.parse(image_dir + '/' + metas[index]).getroot()
		true_bndbox = np.empty(shape = [0,4])
		i = 0
		#true_bndbox['xmin'] = 0
		#true_bndbox['xmax'] = 0
		#true_bndbox['ymin'] = 0
		#true_bndbox['ymax'] = 0
		if meta is not None:
			for obj in meta.findall('object'):
				if obj is not None:
					box = obj.find('bndbox')
					if box is not None:
						true_bndbox = np.append(true_bndbox,[[int(box.find('xmin').text),int(box.find('xmax').text),int(box.find('ymin').text),int(box.find('ymax').text)]],axis=0)
						#true_bndbox['xmin'] = int(box.find('xmin').text)
						#true_bndbox['xmax'] = int(box.find('xmax').text)
						#true_bndbox['ymin'] = int(box.find('ymin').text)
						#true_bndbox['ymax'] = int(box.find('ymax').text)
						i = i + 1
		count = i
		index += 1

		im = cv2.imread(image_dir + '/' + image, cv2.IMREAD_COLOR)
		im = cv2.resize(im, (square_size, square_size))
		im = im.reshape((1, square_size, square_size, 3))

		outputs = predictor([im])

		im = cv2.imread(image_dir + '/' + image, cv2.IMREAD_COLOR)

		objdetect = outputs[0]
		bndboxes  = outputs[1]

		count_bndbox = 0
		max_pred = -100
		max_h = []
		max_w = []
		true_max_h  = []
		true_max_w  = []
		max_h_er1   = []
		max_w_er1   = []
		max_h_er2   = []
		max_w_er2   = []
		for h in range(0, objdetect.shape[1]):
			for w in range(0, objdetect.shape[2]):
				if objdetect[0, h, w] > 0:
					max_h.append(h)
					max_w.append(w)
		true_max_h = list(set(max_h))
		true_max_w = list(set(max_w))
##########
		max_pred = -100
		for i in true_max_h:
			for h in max_h:
				for w in max_w:
					if(h == i):
						if objdetect[0,h,w] > max_pred:
							max_pred = objdetect[0,h,w]
							max_h_i  = h
							max_h_j  = w
			max_h_er1.append(max_h_i)
			max_w_er1.append(max_h_j)

##########
		max_pred = -100
		for j in true_max_w:
			for h in max_h:
				for w in max_w:
					if (w == j):
						if objdetect[0,h,w] > max_pred:
							max_pred = objdetect[0,h,w]
							max_h_fuck1  = h
							max_h_fuck2  = w
			max_h_er2.append(max_h_fuck1)
			max_w_er2.append(max_h_fuck2)

		max_h_er1.extend(max_h_er2)
		max_w_er1.extend(max_w_er2)
		sum_labels= 0;
		bndbox = {}
		bndbox['xmin'] = 0
		bndbox['ymin'] = 0
		bndbox['xmax'] = 0
		bndbox['ymax'] = 0

		bndbox2 =  np.empty(shape = [0,4])   
		for i in range(len(max_w_er1)):
			bndbox2 = np.append(bndbox2,[[int( bndboxes[0,max_h_er1[i],max_w_er1[i],0] + max_w_er1[i]*down_sample_factor),int( bndboxes[0,max_h_er1[i],max_w_er1[i],2] + max_w_er1[i]*down_sample_factor),int( bndboxes[0,max_h_er1[i],max_w_er1[i],1] + max_h_er1[i]*down_sample_factor),int( bndboxes[0,max_h_er1[i],max_w_er1[i],3] + max_h_er1[i]*down_sample_factor)]],axis=0)
			bndbox2[i,0] = int(bndbox2[i,0]*(IMAGE_WIDTH/square_size))
			bndbox2[i,1] = int(bndbox2[i,1]*(IMAGE_WIDTH/square_size))
			bndbox2[i,2] = int(bndbox2[i,2]*(IMAGE_HEIGHT/square_size))
			bndbox2[i,3] = int(bndbox2[i,3]*(IMAGE_HEIGHT/square_size))
			cv2.rectangle(im, (int(bndbox2[i,0]), int(bndbox2[i,2])), (int(bndbox2[i,1]),int(bndbox2[i,3])), (0,255,0), 2)
		print('----------------------------------------')
		#print(str(max_h))
		'''
		print('xmin: ' + str(bndbox2['xmin']))
		print('xmax: ' + str(bndbox2['xmax']))
		print('ymin: ' + str(bndbox2['ymin']))
		print('ymax: ' + str(bndbox2['ymax']))
		'''
		print('sum_labels: ' + str(sum_labels))
		print('truexmin: ' + str(true_bndbox[0,0]))
		print('truexmax: ' + str(true_bndbox[0,1]))
		print('trueymin: ' + str(true_bndbox[0,2]))
		print('trueymax: ' + str(true_bndbox[0,3]))
	    #output_names = ['objdetect_out', 'bndbox_out']
		#for i in range(count):
		#cv2.rectangle(im, (int(max_w*grid_size*(IMAGE_WIDTH/resize_width)),int(max_h*grid_size*(IMAGE_HEIGHT/resize_height))), (int((max_w+1)*grid_size*(IMAGE_WIDTH/resize_width)),int((max_h+1)*grid_size*(IMAGE_HEIGHT/resize_height))), (0,0,255), 1)
		#	cv2.rectangle(im, (int(true_bndbox[i,0]), int(true_bndbox[i,2])), (int(true_bndbox[i,1]), int(true_bndbox[i,3])), (255,0,0), 2)
		#cv2.rectangle(im, (bndbox2['xmin'], bndbox2['ymin']), (bndbox2['xmax'],bndbox2['ymax']), (0,255,0), 2)

		cv2.imshow('image', im)
		cv2.imwrite('images_log/' + image, im)
		cv2.waitKey(800)

def run_single_image(model, sess_init, image):
	print('Running single image!')

	if MONITOR == 1:
		monitor_names = ['conv_class_out', 'image_out', 'conv1_out', 'pool1_out', 'fire1_out', 'pool2_out', 'pool3_out', 'fire5_out', 'fire6_out', 'fire7_out']
	else:
		monitor_names = []
	output_names = ['objdetect_out', 'bndbox_out']
	output_names.extend(monitor_names)

	pred_config = PredictConfig(
		model=model,
		session_init=sess_init,
		input_names=['input'],
		output_names=output_names
	)
	predictor = OfflinePredictor(pred_config)

	if REAL_IMAGE == 1:
		im = cv2.imread(image, cv2.IMREAD_COLOR)
		im = cv2.resize(im, (square_size, square_size))
		cv2.imwrite('test_image.png', im)
		im = im.reshape((1, square_size, square_size, 3))
	else:
		im = np.zeros((1, square_size, square_size, 3))
		k = 0
		for h in range(0, square_size):
			for w in range(0,square_size):
				for c in range (0,3):
					# im[0][h][w][c] = 0
					im[0][h][w][c] = k%256
					k += 1

	outputs = predictor([im])

	objdetect = outputs[0]
	bndboxes = outputs[1]

	max_pred = -100
	max_h = -1
	max_w = -1
	for h in range(0, objdetect.shape[1]):
		for w in range(0, objdetect.shape[2]):
			if objdetect[0, h, w] > max_pred:
				max_pred = objdetect[0, h, w]
				max_h = h
				max_w = w
	bndbox2 = {}
	bndbox2['xmin'] = int( bndboxes[0,max_h,max_w,0] + max_w*down_sample_factor)
	bndbox2['ymin'] = int( bndboxes[0,max_h,max_w,1] + max_h*down_sample_factor)
	bndbox2['xmax'] = int( bndboxes[0,max_h,max_w,2] + max_w*down_sample_factor)
	bndbox2['ymax'] = int( bndboxes[0,max_h,max_w,3] + max_h*down_sample_factor)
	bndbox2['xmin'] = int(bndbox2['xmin']*(640/square_size))
	bndbox2['ymin'] = int(bndbox2['ymin']*(360/square_size))
	bndbox2['xmax'] = int(bndbox2['xmax']*(640/square_size))
	bndbox2['ymax'] = int(bndbox2['ymax']*(360/square_size))

	im = cv2.imread(image, cv2.IMREAD_COLOR)
	cv2.rectangle(im, (bndbox2['xmin'], bndbox2['ymin']), (bndbox2['xmax'],bndbox2['ymax']), (0,255,0), 2)
	cv2.imshow('image', im)
	cv2.waitKey(2000)

	print('max_h: ' + str(max_h))
	print('max_w: ' + str(max_w))
	print('objdetect: ' + str(objdetect))
	print('bndboxes: ' + str(bndboxes[0,max_h,max_w]))
	
	index = 2
	for o in monitor_names:
		print(o + ', shape: ' + str(outputs[index].shape) )

		if 'image' not in o:
			print(str(outputs[index]))

		if len(outputs[index].shape) == 4:
			file_name = o.split('/')[-1]
			print('Writing file... ' + file_name)
			if not os.path.exists('./log'):
				os.makedirs('./log')
			with open('./log/' + file_name + '.log', 'w') as f:
				for sample in range(0, outputs[index].shape[0]):
					for h in range(0, outputs[index].shape[1]):
						for w in range(0, outputs[index].shape[2]):
							res = ''
							for c in range(0, outputs[index].shape[3]):
								if 'image' in file_name:
									res = hexFromInt( int(outputs[index][sample, h, w, c]), 8 ) + '_' + res
								elif 'noact' in file_name:
									temp = (2**FACTOR_SCALE_BITS)*outputs[index][sample, h, w, c]
									res = hexFromInt( int(temp), 32 ) + '_' + res
								else:
									res = hexFromInt( int(outputs[index][sample, h, w, c]), BITA) + '_' + res
							f.write('0x' + res + '\n')
		index += 1

def dump_weights(meta, model, output):
	fw, fa, fg = get_dorefa(BITW, BITA, BITG)

	with tf.Graph().as_default() as G:
		tf.train.import_meta_graph(meta)

		init = get_model_loader(model)
		sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
		sess.run(tf.global_variables_initializer())
		init.init(sess)

		with sess.as_default():
			if output:
				if output.endswith('npy') or output.endswith('npz'):
					varmanip.dump_session_params(output)
				else:
					var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
					var.extend(tf.get_collection(tf.GraphKeys.MODEL_VARIABLES))
					var_dict = {}
					for v in var:
						name = varmanip.get_savename_from_varname(v.name)
						var_dict[name] = v
					logger.info("Variables to dump:")
					logger.info(", ".join(var_dict.keys()))
					saver = tf.train.Saver(
						var_list=var_dict,
						write_version=tf.train.SaverDef.V2)
					saver.save(sess, output, write_meta_graph=False)

			network_model = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
			network_model.extend(tf.get_collection(tf.GraphKeys.MODEL_VARIABLES))

			target_frequency = 200000000
			target_FMpS = 300
			non_quantized_layers = ['conv1/Conv2D', 'conv_obj/Conv2D', 'conv_box/Conv2D']

			json_out, layers_list, max_cycles = generateLayers(sess, BITA, BITW, non_quantized_layers, target_frequency, target_FMpS)
			
			achieved_FMpS = target_frequency/max_cycles

			if DEMO_DATASET == 0:
				generateConfig(layers_list, 'halfsqueezenet-config.h')
				genereateHLSparams(layers_list, network_model, 'halfsqueezenet-params.h', fw)
			else:
				generateConfig(layers_list, 'halfsqueezenet-config_demo.h')
				genereateHLSparams(layers_list, network_model, 'halfsqueezenet-params_demo.h', fw)

			print('|---------------------------------------------------------|')
			print('target_FMpS: ' + str(target_FMpS) )
			print('achieved_FMpS: ' + str(achieved_FMpS) )

if __name__ == '__main__':
	print('Start')

	parser = argparse.ArgumentParser()
	parser.add_argument('dump2_train1_test0', help='dump(2), train(1) or test(0)')
	parser.add_argument('--model', help='model file')
	parser.add_argument('--meta', help='metagraph file')
	parser.add_argument('--output', help='output for dumping')
	parser.add_argument('--gpu', help='the physical ids of GPUs to use')
	parser.add_argument('--data', help='DAC dataset dir')
	parser.add_argument('--run', help='directory of images to test')
	parser.add_argument('--weights', help='weights file')
	args = parser.parse_args()

	print('Using GPU ' + str(args.gpu))

	if args.gpu:
		os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

	print(str(args.dump2_train1_test0))

	if args.dump2_train1_test0 == '1':
		if args.data == None:
			print('Provide DAC dataset path with --data')
			sys.exit()

		config = get_config()
		if args.model:
			config.session_init = SaverRestore(args.model)

		SimpleTrainer(config).train()

	elif args.dump2_train1_test0 == '0':
		if args.run == None:
			print('Provide images with --run ')
			sys.exit()
		if args.weights == None:
			print('Provide weights file (.npy) for testing!')
			sys.exit()

		assert args.weights.endswith('.npy')
		run_image(Model(), DictRestore(np.load(args.weights, encoding='latin1').item()), args.run)

	elif args.dump2_train1_test0 == '2':
		if args.meta == None:
			print('Provide meta file (.meta) for dumping')
			sys.exit()
		if args.model == None:
			print('Provide model file (.data-00000-of-00001) for dumping')
			sys.exit()

		dump_weights(args.meta, args.model, args.output)

	elif args.dump2_train1_test0 == '3':
		if args.run == None:
			print('Provide image with --run ')
			sys.exit()
		if args.weights == None:
			print('Provide weights file (.npy) for testing!')
			sys.exit()

		assert args.weights.endswith('.npy')
		run_single_image(Model(), DictRestore(np.load(args.weights, encoding='latin1').item()), args.run)