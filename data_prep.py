import pandas as pd
import numpy as np
import datetime
from dateutil import parser
import helpers
from tqdm import tqdm

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

	for activity in tqdm(parsed_activities,desc='finding max and min'):
		for sensor_activity in activity:
			sensor_id,sensor_name,activity_name,sensor_dt_range,activity_dt_range = sensor_activity
			sensors.append(sensor_id)	
			dts.append(min(sensor_dt_range))
			dts.append(max(sensor_dt_range))
			dts.append(min(activity_dt_range))
			dts.append(max(activity_dt_range))
	
	num_channels = np.unique(sensors)

	min_time = min(dts)
	min_time = min_time.round('D')

	max_time = max(dts)
	max_time = max_time.round('D')

	time_index = pd.date_range(min_time,max_time,freq='1S')
	cols = num_channels.tolist()+['activity']
	df = pd.DataFrame(0,index=time_index,columns=cols)
	df.index.name = 'time'
	df.loc[:,'activity'] = 'no_activity'

	for activity in tqdm(parsed_activities,desc='merging on time axis'):
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
	# for activity in tqdm(activities,desc='parsing activities'):
	# 	parsed_activity = parse_activity(activity)
	# 	parsed_activities.append(parsed_activity)

	# parsed_activities = np.array(parsed_activities)

	# helpers.clear_folder('tmp/')
	# np.savez_compressed('tmp/sub1_parsed_activities.npz',activity=parsed_activities)

	# # # SAVE in time vs sensor format

	# parsed_activities = np.load('tmp/sub1_parsed_activities.npz')['activity']
	# df = merge_on_time_axis(parsed_activities)
	# df.index = pd.to_datetime(df.index)
	# df['day'] = df.index.day

	# helpers.clear_folder('data/prep_data')
	# df.to_csv('data/prep_data/sub1_gzip.csv',compression='gzip')
	# df.to_csv('data/prep_data/sub1_uncompressed.csv')

	df = pd.read_csv('data/prep_data/sub1_gzip.csv',compression='gzip')
	# print(df)
	print(df.columns)