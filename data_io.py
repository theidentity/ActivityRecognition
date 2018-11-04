import numpy as np
import pandas as pd
from sklearn import preprocessing
from scipy import stats
from sklearn.model_selection import train_test_split

def split_windows(x,y,time_window,filter_class=6):

	y = y.reshape(-1,time_window)
	y = stats.mode(y,axis=1)[0]
	y = y.flatten()

	num_splits = len(x)//time_window
	x = np.array(np.vsplit(x,num_splits))

	idx = y!= filter_class
	idx = idx.flatten()
	x = x[idx,:,:]
	y = y[idx]

	return x,y

def get_windowed_data(time_window,hold_out,use_saved,show_stats,replace_cat=True):

	dates = list(range(27,32)) + list(range(1,12))

	if not use_saved:
		df = pd.read_csv('data/prep_data/sub1_gzip.csv',compression='gzip')
		df = df.iloc[:-1,:]

		nine_cat = {

			'Bathing' : 'Personal needs',
			'Cleaning': 'Domestic work',
			'Doing laundry': 'Domestic work',
			'Dressing': 'Personal needs',

			'Going out for entertainment':'Entertainment and social life',
			'Going out for shopping':'Purchasing goods and services',

			'Going out to work' : 'Employment',
			'Grooming': 'Personal needs',
			'Lawnwork': 'Domestic work',
			'Other' : 'Other',

			'Preparing a beverage': 'Domestic work',
			'Preparing a snack': 'Domestic work',
			'Preparing breakfast': 'Domestic work',

			'Preparing dinner': 'Domestic work',
			'Preparing lunch': 'Domestic work',
			'Putting away dishes': 'Domestic work',

			'Putting away groceries': 'Domestic work',
			'Putting away laundry': 'Domestic work',
			'Toileting': 'Personal needs',

			'Washing dishes': 'Domestic work',
			'Washing hands': 'Personal needs',
			'Watching TV': 'Personal needs',
			'no_activity' : 'no_activity' 
		}

		if replace_cat:
			df['activity'] = df['activity'].replace(nine_cat)

		x = df.iloc[:,1:-2].values
		y = df.iloc[:,-2].values
		day = df.iloc[:,-1].values

		print(np.unique(y))
		le = preprocessing.LabelEncoder()
		y = le.fit_transform(y)
		print(np.unique(y,return_counts=True))

		np.savez_compressed('data/sub1_label_encoded.npz',x=x,y=y,day=day)

	data = np.load('data/sub1_label_encoded.npz')
	x,y,day = data['x'],data['y'],data['day']

	if hold_out in dates:
		idx = day!=hold_out
		train_x,train_y = x[idx],y[idx]
		idx = day==hold_out
		test_x,test_y = x[idx],y[idx]
		train_x,train_y = split_windows(train_x,train_y,time_window=time_window,filter_class=6)
		test_x,test_y = split_windows(test_x,test_y,time_window=time_window,filter_class=6)
	else:
		x,y = split_windows(x,y,time_window=time_window,filter_class=6)
		train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.20, random_state=42)


	if show_stats:
		print(train_x.shape)
		print(train_y.shape)
		print(np.unique(train_y,return_counts=True))
		
		print(test_x.shape)
		print(test_y.shape)
		print(np.unique(test_y,return_counts=True))

	return train_x,train_y,test_x,test_y


if __name__ == '__main__':

	get_windowed_data(time_window=5,hold_out=31,use_saved=True,show_stats=True)
	get_windowed_data(time_window=5,hold_out=-1,use_saved=True,show_stats=True)
