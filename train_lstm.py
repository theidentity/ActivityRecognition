seed = 42

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
np.random.seed(seed)
from sklearn import preprocessing
import keras
from keras.utils import to_categorical
from keras.layers import Dense,Input,Flatten,Dropout
from keras.layers import Embedding,LSTM
from keras.models import Model,load_model
from keras import callbacks
from sklearn import metrics
from sklearn.utils.class_weight import compute_class_weight

import data_io


class ActivityRecognizer(object):
	"""docstring for ActivityRecognizer"""
	def __init__(self, hold_out_day):
		
		self.hold_out_day = hold_out_day
		self.batch_size = 100
		self.num_classes = 6

		self.time_window = 5
		self.num_features = 72
		self.input_shape = (self.time_window,self.num_features)

		self.name = 'activity_reco_lstm'+str(hold_out_day)
		self.save_path = 'models/'+self.name+'.h5'
		self.train_log_path = 'logs/train_log_'+self.name+'.csv'
		self.save_metrics_path = 'logs/results_'+self.name+'.txt'
		self.cws = None

		print(self.name)

	def load_data(self,show_stats=False):

		train_x,train_y,valid_x,valid_y = data_io.get_windowed_data(time_window=self.time_window,hold_out=self.hold_out_day,
			use_saved=True,show_stats=show_stats)

		self.cws = compute_class_weight(class_weight='balanced', classes=np.unique(train_y), y=train_y)
		print('Class Weights',self.cws)

		train_y = to_categorical(train_y,num_classes=self.num_classes).astype(np.uint8)
		valid_y = to_categorical(valid_y,num_classes=self.num_classes).astype(np.uint8)

		# print(np.sum(train_y,axis=0))
		# print(np.sum(valid_y,axis=0))

		print(train_x.shape)
		print(train_y.shape)

		return train_x,train_y,valid_x,valid_y

	def get_model(self):

		inp = Input(shape=self.input_shape)
		x = inp
		# x = Embedding(2500,500,input_length=self.time_window,dropout=.2)(x)
		# print(x.shape)
		x = LSTM(2000,dropout_U=.2,dropout_W=.2)(x)
		pred = Dense(self.num_classes,activation='softmax')(x)
		model = Model(inp,pred)
		return model

	def build_model(self,lr):
		model = self.get_model()
		model.compile(
			loss = 'categorical_crossentropy',
			optimizer = keras.optimizers.Adam(lr),
			metrics = ['accuracy']
			)

		model.summary()
		return model

	def get_callbacks(self):

		checkpointer = callbacks.ModelCheckpoint(filepath=self.save_path, monitor='val_loss', verbose=1, save_best_only=True)
		early_stopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-7, patience=15, verbose=1, mode='auto')
		reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)
		csv_logger = callbacks.CSVLogger(self.train_log_path, separator=',', append=False)
		# tensorboard = TensorBoard(log_dir='logs/', histogram_freq=0, batch_size=self.batch_size, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
		return [early_stopping,checkpointer,reduce_lr,csv_logger]


	def train_model(self,num_epochs,lr):

		model = self.build_model(lr)

		train_x,train_y,valid_x,valid_y = self.load_data()
		model.fit(x=train_x,
			y=train_y,
			batch_size=self.batch_size,
			epochs=num_epochs,
			verbose=1,
			callbacks=self.get_callbacks(),
			validation_data=(valid_x,valid_y),
			shuffle=True,
			# class_weight=self.cws,
			initial_epoch=0,
			steps_per_epoch=None,
			validation_steps=None
			)

	def get_pred(self,use_saved):

		if not use_saved:
			__,__,valid_x,valid_y = self.load_data()
			model = load_model(self.save_path)

			y_pred_prob = model.predict(valid_x,batch_size=self.batch_size,verbose=1)
			np.savez_compressed('tmp/'+self.name+'.npz',y_true=valid_y,y_pred_prob=y_pred_prob)
			return valid_y,y_pred_prob
		else:
			data = np.load('tmp/'+self.name+'.npz')
			return data['y_true'],data['y_pred_prob']


	def evaluate(self,use_saved):

		y_true,y_pred_prob = self.get_pred(use_saved)
		
		aucs = {}

		for i in range(self.num_classes):
			y_cl_true = y_true[:,i]
			y_cl_pred_prob = y_pred_prob[:,i]
			if np.sum(y_cl_true) > 0 : 
				auc_score = metrics.roc_auc_score(y_cl_true,y_cl_pred_prob)
				aucs[i] = auc_score
			else:
				aucs[i] = 0.0
			# print(aucs)

		y_true = np.argmax(y_true,axis=1)
		y_pred = np.argmax(y_pred_prob,axis=1)

		avg_auc = [aucs[x] for x in aucs.keys()]
		avg_auc = np.mean(avg_auc)

		cm = metrics.confusion_matrix(y_true,y_pred)
		acc = metrics.accuracy_score(y_true,y_pred)

		file = open(self.save_metrics_path,'w+')
		for item in [cm,acc,aucs,avg_auc]:
			print(item)
			print(item,file=file)

		file.close()

		aucs = [aucs[x] for x in range(self.num_classes)]
		return aucs


if __name__ == '__main__':

	# dates =  list(range(-1,0)) + list(range(27,32)) + list(range(1,12))
	# aucs_set = []

	# for i in dates:
	# 	clf = ActivityRecognizer(hold_out_day=i)
	# 	# clf.load_data()
	# 	clf.train_model(num_epochs=100,lr=1e-2)
	# 	aucs = clf.evaluate(use_saved=False)
	# 	aucs_set.append(aucs)

	# aucs_set = np.array(aucs_set)
	# np.save('tmp/aucs_set.npy',aucs_set)

	aucs_set = np.load('tmp/aucs_set.npy')
	print(aucs_set)
	print(np.average(aucs_set, axis=0, weights=aucs_set.astype(bool)))


	# clf = ActivityRecognizer(hold_out_day=-1)
	# # clf.load_data(show_stats=True)
	# clf.train_model(num_epochs=100,lr=1e-2)
	# aucs = clf.evaluate(use_saved=False)
	# print(aucs)