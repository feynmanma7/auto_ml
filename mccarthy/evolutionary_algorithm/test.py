import pickle

if __name__ == '__main__':
	my_dict = {}
	with open("/Users/flyingman/Data/tmp.txt", 'rb') as fr:
		try:
			my_dict = pickle.load(fr)
		except:
			import traceback
			traceback.print_exc()

	print(my_dict)