import numpy as np
import pandas as pd


def read_data():
	df = pd.read_csv('data/prep_data/sub1.csv',compression='gzip')
	df['time'] = pd.to_datetime(df['time'])
	df['date'] = df['time'].dt.date
	return df

def get_train_test():
	df = read_data()
	unique_dates = np.unique(df['date'])
	train_dates = unique_dates[:-4]
	test_dates = unique_dates[-4:]
	print(train_dates)
	print(test_dates)

	train_df = df[df['date'].isin(train_dates)]
	test_df = df[df['date'].isin(test_dates)]

	print(train_df.shape)
	print(test_df.shape)

	print(len(np.unique(train_df['activity'])))
	print(len(np.unique(test_df['activity'])))
	
	print(np.unique(train_df['activity'],return_counts=True))
	print(np.unique(test_df['activity'],return_counts=True))
	return train_df,test_df

def split_to_time_windows(df):

	rows = df.shape[0]
	window_size = 15
	inds = np.arange(0,rows,window_size)

	# dfs = np.split(df,inds,axis=1)

	for idx in inds:
		df_win = df.loc[idx:idx+window_size,:]
		activity = df_win['activity']
		unique = np.unique(activity)
		unique = np.setdiff1d(unique,['no_activity'])
		if len(unique) > 1 :
			print(unique)



if __name__ == '__main__':
	df = read_data()
	# get_train_test()
	split_to_time_windows(df)