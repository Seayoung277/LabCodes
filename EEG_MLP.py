import tensorflow as tf
import numpy as np
import scipy.io as sio
import random
import sys

def readData(dataPath, labelPath):
	dataFile = sio.loadmat(dataPath)
	labelFile = sio.loadmat(labelPath)
	data = dataFile['X'][0][0]
	label = labelFile['Y'][0][0]
	for i in range(14):
		data = np.concatenate((data, dataFile['X'][0][i+1]))
		label = np.concatenate((label, labelFile['Y'][0][i+1]))

	return data, label

def normalize(data):
	for i in range(data.shape[1]):
		dataMean = data[:][i].mean()
		dataStd = data[:][i].std()
		for j in range(data.shape[0]):
			data[j][i] = (data[j][i] - dataMean)/dataStd

	return data

def getBatch(data, label, size, index=0):
	batchData = [data[index]]
	batchLabel = [label[index]]
	for i in range(size-1):
		index = (index + 1) % data.shape[0]
		batchData = np.concatenate((batchData, [data[index]]))
		batchLabel = np.concatenate((batchLabel, [label[index]]))

	return batchData, batchLabel

def main(argv):

	normFlag = argv[1]
	holdSeed = argv[2]
	seed = 10086
	hid1 = 256
	hid2 = 128
	learningRate = 0.001
	iterNum = 1000000
	batchSize = 100

	dataPath = '../Data/EEG/EEG_X.mat'
	labelPath = '../Data/EEG/EEG_Y.mat'
	logPath = '../Log/EEG'
	data, label = readData(dataPath, labelPath)

	if(normFlag):
		data = normalize(data)

	if(holdSeed):
		tf.set_random_seed(seed)

	for i in range(15):
		testData = data[(i*3394):((i+1)*3394-1)][:]
		testLabel = label[(i*3394):((i+1)*3394-1)][:]
		trainData = np.delete(data, range(i*3394,(i+1)*3394-1), axis=0)
		trainLabel = np.delete(label, range(i*3394,(i+1)*3394-1), axis=0)

		x = tf.placeholder(tf.float32, shape=[None, 310])
		y = tf.placeholder(tf.float32, shape=[None, 1])
		w1 = tf.Variable(tf.truncated_normal([310, hid1], stddev=0.1))
		w2 = tf.Variable(tf.truncated_normal([hid1, hid2], stddev=0.1))
		w3 = tf.Variable(tf.truncated_normal([hid2, 3], stddev=0.1))
		b1 = tf.Variable(tf.zeros(hid1))
		b2 = tf.Variable(tf.zeros(hid2))
		b3 = tf.Variable(tf.zeros(3))
		p1 = tf.nn.relu(tf.add(tf.matmul(x, w1), b1))
		p2 = tf.nn.relu(tf.add(tf.matmul(p1, w2), b2))
		a = tf.nn.softmax(tf.add(tf.matmul(p2, w3), b3))
		loss = tf.nn.l2_loss(tf.subtract(tf.one_hot(tf.to_int32(y), 3), a))
		optimizer = tf.train.AdamOptimizer()
		train = optimizer.minimize(loss)
		accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(a, 1), tf.to_int64(y)), tf.float64))

		init = tf.global_variables_initializer()
		sess = tf.Session()
		sess.run(init)
		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
		sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))        

		#tf.summary.scalar('Loss', loss)
		#tf.summary.scalar('Accuracy', accuracy)
		#summary = tf.summary.merge_all()
		#summary_writer = tf.summary.FileWriter(logPath, sess.graph)

		for e in range(iterNum):
			batchData, batchLabel = getBatch(trainData, trainLabel, batchSize)
			_, batchLoss = sess.run([train, loss], feed_dict={x:batchData, y:batchLabel})
			batchAccuracy = sess.run(accuracy, feed_dict={x:testData, y:testLabel})
			#summary_writer.add_summary(batchSummary, e)
			print('Iter %d Loss %f Acc %f' %(e, batchLoss, batchAccuracy))

		sess.close()
if __name__ == '__main__':
    main(sys.argv)