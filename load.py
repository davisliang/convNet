import glob
import os
import theano
from theano import tensor as T
import numpy
from PIL import Image

def loadData(path):

	"""

	reads and preprocesses data in path of format:

	path = '/Users/Davis/Desktop/dataset'

	"""


	newSetData = [] #create the original data list. Right now it's just a []
	newSetLabels = []	#create the original label list

	for files in glob.glob(path):
		fileName = os.path.split(files)[1][0:-4] #extract filename without jpg

		label = fileName[0]	#this is the label for the image
		img = Image.open(files) #load image
		img_grey = img.convert('L') #greyscales


		np_img_grey = numpy.asarray(img_grey.getdata(),dtype=numpy.float32)/255	#makes  greyscale images into numpy arrays with magnitude of 1
		np_img_grey_shaped = np_img_grey.reshape((img_grey.size[1], img_grey.size[0]))	#shapes the numpy arrays
		imgdata = numpy.resize(np_img_grey, (100,100)) #final data is resized and ready to use
		imgdata_reshaped = imgdata.reshape(10000)	#put into a vector for easy carrying
		imgdata_list = numpy.ndarray.tolist(imgdata_reshaped)	#now its a list
		newSetData.append(imgdata_list)	#creates list of numpy arrays

		newSetLabels.append(label)

		#use len(newSetData) to get the LENGTH (zero index) of the dataset
		#your values are accessed with 
		#  
		#
		#		array[image_number][row number, column number] (zero index)

	newSetData = numpy.asarray(newSetData, dtype=numpy.float32)	#turn them back into numpy arrays
	newSetLabels = numpy.asarray(newSetLabels, dtype=numpy.int)	#turn them back into numpy arrays

	data = (newSetData, newSetLabels)

	def shared_dataset(data_xy, borrow=True):
		""" Function that loads the dataset into shared variables

		The reason we store our dataset in shared variables is to allow
		Theano to copy it into the GPU memory (when code is run on GPU).
		Since copying data into the GPU is slow, copying a minibatch everytime
		is needed (the default behaviour if the data is not in a shared
		variable) would lead to a large decrease in performance.
		"""
		data_x, data_y = data_xy
		shared_x = theano.shared(numpy.asarray(data_x,
											   dtype=theano.config.floatX),
								 borrow=borrow)
		shared_y = theano.shared(numpy.asarray(data_y,
											   dtype=theano.config.floatX),
								 borrow=borrow)
		# When storing data on the GPU it has to be stored as floats
		# therefore we will store the labels as ``floatX`` as well
		# (``shared_y`` does exactly that). But during our computations
		# we need them as ints (we use labels as index, and if they are
		# floats it doesn't make sense) therefore instead of returning
		# ``shared_y`` we will have to cast it to int. This little hack
		# lets ous get around this issue
		return shared_x, T.cast(shared_y, 'int32')

	dataSet, labelSet = shared_dataset(data)
	finalData = (dataSet, labelSet)
	return finalData
	
if __name__ == '__main__':
	finalData = loadData('/Users/Davis/Desktop/dataset/train/*.jpg') #test set...
	print finalData






