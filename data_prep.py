import pandas as pd
import numpy as np
import datetime
from dateutil import parser
import helpers


sampling_freq = '1s'

def read_data():
	path = 'data/subject1/activities_data.csv'
	data = open(path,'r').read()
	return data

def split_activities(data):
	lines = data.split('\n')
	lines = np.array(lines)

	# leave last blank line
	lines = lines[:-1]

	# split activities
	activity_idx = np.arange(0,len(lines),5)
	activities = np.array([lines[x:x+5] for x in activity_idx])
	return activities

def create_datetime_range(date,start_time,end_time):
	
	date = parser.parse(date)
	start_time = parser.parse(start_time)
	end_time = parser.parse(end_time)
	
	start_time = datetime.datetime.combine(date.date(),start_time.time())
	end_time = datetime.datetime.combine(date.date(),end_time.time())
	datetime_range = pd.date_range(start_time,end_time,freq='1S')

	return datetime_range

def parse_activity(activity):
	l1 = activity[0]

	activity_name,date,start_time,end_time = l1.split(',')
	activity_dt_range = create_datetime_range(date,start_time,end_time)

	l2 = activity[1].split(',')
	l3 = activity[2].split(',')
	l4 = activity[3].split(',')
	l5 = activity[4].split(',')

	parsed_activitiy = []

	for idx,l2 in enumerate(l2):
		sensor_id = l2
		sensor_name = l3[idx]
		start_time = l4[idx]
		end_time = l5[idx]

		sensor_dt_range = create_datetime_range(date,start_time,end_time)
		parsed_activitiy.append([sensor_id,sensor_name,activity_name,sensor_dt_range,activity_dt_range])

	return parsed_activitiy

def merge_on_time_axis(parsed_activities):

	dts = []
	sensors = []

	for i,activity in enumerate(parsed_activities):
		print(i)
		for sensor_activity in activity:
			sensor_id,sensor_name,activity_name,sensor_dt_range,activity_dt_range = sensor_activity
			sensors.append(sensor_id)	
			dts.append(min(sensor_dt_range))
			dts.append(max(sensor_dt_range))
			dts.append(min(activity_dt_range))
			dts.append(max(activity_dt_range))

	num_channels = np.unique(sensors)
	time_index = pd.date_range(min(dts),max(dts),freq='1S')
	cols = num_channels.tolist()+['activity']
	df = pd.DataFrame(0,index=time_index,columns=cols)
	df.index.name = 'time'
	df.loc[:,'activity'] = 'no_activity'

	for i,activity in enumerate(parsed_activities):
		print(i)
		for sensor_activity in activity:
			sensor_id,sensor_name,activity_name,sensor_dt_range,activity_dt_range = sensor_activity
			df.loc[sensor_dt_range,sensor_id] = 1
			df.loc[activity_dt_range,'activity'] = activity_name

	return df

if __name__ == '__main__':

	# PARSE RAW activities

	# data = read_data()
	# activities = split_activities(data)
	# parsed_activities = []
	# for activity in activities:
	# 	parsed_activity = parse_activity(activity)
	# 	parsed_activities.append(parsed_activity)

	# parsed_activities = np.array(parsed_activities)

	# helpers.clear_folder('tmp/')
	# np.savez_compressed('tmp/sub1_parsed_activities.npz',activity=parsed_activities)

	# # SAVE in time vs sensor format

	# parsed_activities = np.load('tmp/sub1_parsed_activities.npz')['activity']
	# df = merge_on_time_axis(parsed_activities[:])
	# df.to_csv('tmp/sub1.csv',compression='gzip')

	# Save compressed numpy arrays

	df = pd.read_csv('tmp/sub1.csv',compression='gzip')
	cols = df.columns
	time = df.iloc[:,0]
	X = df.iloc[:,1:-1].values
	y = df.iloc[:,-1].values

	helpers.clear_folder('data/prep_data/')
	save_path = 'data/prep_data/sub1.npz'
	np.savez_compressed(save_path,x=X,y=y,col_names=cols,time=time)
	print('Compressed file saved to :',save_path)