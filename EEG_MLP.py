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
		label = np.concatenate((label, labelFile['Y'][0][i+1]), axis=1)
	label = label + np.ones(label.shape, dtype=np.int)

	return data, label[0]

def normalize(data):
	for i in range(data.shape[1]):
		dataMean = data[:][i].mean()
		dataStd = data[:][i].std()
		for j in range(data.shape[0]):
			data[j][i] = (data[j][i] - dataMean)/dataStd

	return data

def getBatch(data, label, size):
	batchData = [data[random.randint(0, data.shape[0]-1)]]
	batchLabel = [label[random.randint(0, data.shape[0]-1)]]
	for i in range(size-1):
		batchData = np.concatenate((batchData, [data[random.randint(0, data.shape[0]-1)]]))
		batchLabel = np.concatenate((batchLabel, [label[random.randint(0, data.shape[0]-1)]]))

	return batchData, batchLabel

def main(argv):

	normFlag = int(argv[1])
	holdSeed = int(argv[2])
	seed = 10086
	hid1 = 256
	hid2 = 256
	learningRate = 0.0001
	momentum = 0.9
	iterNum = 2000
	batchSize = 1000

	dataPath = '../Data/EEG/EEG_X.mat'
	labelPath = '../Data/EEG/EEG_Y.mat'
	logPath = '../Log/EEG'
	data, label = readData(dataPath, labelPath)

	if(normFlag):
		data = normalize(data)

	if(holdSeed):
		tf.set_random_seed(seed)
		random.seed(seed)

	accAll = []

	for i in range(15):
		testData = data[(i*3394):((i+1)*3394-1)][:]
		testLabel = label[(i*3394):((i+1)*3394-1)]
		trainData = np.delete(data, range(i*3394,(i+1)*3394-1), axis=0)
		trainLabel = np.delete(label, range(i*3394,(i+1)*3394-1))
		print(data.shape, np.max(data))
		print(label.shape, np.max(label))
		print(trainData.shape, np.max(trainData))
		print(trainLabel.shape, np.max(trainLabel))
		print(testData.shape, np.max(testData))
		print(testLabel.shape, np.max(testLabel))

		x = tf.placeholder(tf.float32, shape=[None, 310])
		y = tf.placeholder(tf.int32, shape=[None])
		w1 = tf.Variable(tf.truncated_normal([310, hid1], stddev=0.1))
		w2 = tf.Variable(tf.truncated_normal([hid1, hid2], stddev=0.1))
		w3 = tf.Variable(tf.truncated_normal([hid2, 3], stddev=0.1))
		b1 = tf.Variable(tf.zeros(hid1))
		b2 = tf.Variable(tf.zeros(hid2))
		b3 = tf.Variable(tf.zeros(3))
		p1 = tf.nn.relu(tf.add(tf.matmul(x, w1), b1))
		p2 = tf.nn.relu(tf.add(tf.matmul(p1, w2), b2))
		a = tf.add(tf.matmul(p2, w3), b3)
		loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=a))
		optimizer = tf.train.AdamOptimizer()
		train = optimizer.minimize(loss)
		accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(a, 1), tf.to_int64(y)), tf.float64))

		init = tf.global_variables_initializer()
		sess = tf.Session()
		sess.run(init)
		#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
		#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))        

		#tf.summary.scalar('Loss', loss)
		#tf.summary.scalar('Accuracy', accuracy)
		#summary = tf.summary.merge_all()
		#summary_writer = tf.summary.FileWriter(logPath, sess.graph)
		accMax = 0
		print('Training CV %d ...' % i)
		for e in range(iterNum):
			batchData, batchLabel = getBatch(trainData, trainLabel, batchSize)
			_, batchLoss = sess.run([train, loss], feed_dict={x:batchData, y:batchLabel})
			batchAccuracy = sess.run(accuracy, feed_dict={x:testData, y:testLabel})
			#summary_writer.add_summary(batchSummary, e)
			print('Iter %d Loss %f Acc %f' %(e, batchLoss, batchAccuracy))
			if(batchAccuracy>accMax):
				accMax = batchAccuracy
		accAll.append(accMax)
		print('Max Accuracy %f' % accMax)
		sess.close()

	accSum = 0
	for i in range(15):
		accSum += accAll[i]
	print('Overall Accuracy %f' % (accSum/15))

if __name__ == '__main__':
    main(sys.argv)