import tensorflow as tf
import numpy as np
import scipy.io as sio
import argparse as ap
import random
import copy
import sys

def parseArg():
	parser = ap.ArgumentParser(description='Tensorflow Based EEG MLP Classifier')
	parser.add_argument('-l', '--layers', type=int, nargs='+', default=[128,], metavar='L',
		help='List of neural numbers of each layer added to the network (default: [128,])')
	parser.add_argument('-n', '--norm', type=int, default=0, metavar='N',
		help='''Type of normalization to apply (default: None)
		[0]	None
		[1]	Batch
		[2]	L2''')
	parser.add_argument('-o', '--optimizer', type=int, default=0, metavar='O',
		help='''Type of optimizer to apply (default: Gradient Descent)
		[0]	Gradient Descent
		[1]	Adam
		[2]	Adadelta
		[3]	Adagrad
		[4]	RMSProp''')
	parser.add_argument('-ls', '--loss', type=int, default=0, metavar='LS',
		help='''Type of loss function to apply (default: Cross Entropy)
		[0]	Cross Entropy
		[1]	L2 loss''')
	parser.add_argument('-lr', '--learning-rate', type=float, default=0.001, metavar='LR',
		help='Learning rate (default: 0.001)')
	parser.add_argument('-e', '--epoch', type=int, default=100, metavar='E',
		help='Number of epoches to run (default: 100)')
	parser.add_argument('-b', '--batch-size', type=int, default=100, metavar='B',
		help='Batch size (default: 100)')
	parser.add_argument('-r', '--batch-rule', type=int, default=0, metavar='R',
		help='''Betch rule (default: random)
		[0]	Random selection
		[1]	Sequential selection''')
	parser.add_argument('-s', '--random-seed', type=int, default=0, metavar='S',
		help='Random seed (default: random)')
	parser.add_argument('-m', '--memory-usage', type=float, default=0, metavar='M',
		help='Percentage of GPU memory to use (default: no GPU)')
	parser.add_argument('-i', '--info', type=int, default=1, metavar='I',
		help='Whether to show batch loss and accuracy (default 1)')
	parser.add_argument('-c', '--cross-valid', type=int, default=1, metavar='C',
		help='Cross validation (default 1)')
	parser.add_argument('-d', '--drop-out', type=float, default=1., metavar='D',
		help='Dropout Rate (default no drop out)')
	parser.add_argument('-a', '--activ', type=int, default=0, metavar='A',
		help='''Activation function (default relu)
		[0]	relu
		[1]	sigmoid''')

	args = parser.parse_args()

	args.cross_valid -= 1

	if(args.random_seed):
		tf.set_random_seed(args.random_seed)
		random.seed(args.random_seed)
	else:
		tf.set_random_seed(random.randint(0, sys.maxint))
		random.seed(random.randint(0, sys.maxint))

	return args

def readData(dataPath, labelPath):
	dataFile = sio.loadmat(dataPath)
	labelFile = sio.loadmat(labelPath)
	data = dataFile['X'][0][0]
	label = labelFile['Y'][0][0]
	for i in range(14):
		data = np.concatenate((data, dataFile['X'][0][i+1]))
		label = np.concatenate((label, labelFile['Y'][0][i+1]))
	label = label + np.ones(label.shape, dtype=np.int)

	return data, label

def getBatch(data, label, size, rule, index=-1):
	if(rule):
		index = (index+1) % data.shape[0]
	else:
		index = random.randint(0, data.shape[0]-1)
	batchData = [data[index]]
	batchLabel = [label[index]]
	for i in range(size-1):
		if(rule):
			index = (index+1) % data.shape[0]
		else:
			index = random.randint(0, data.shape[0]-1)
		batchData = np.concatenate((batchData, [data[index]]))
		batchLabel = np.concatenate((batchLabel, [label[index]]))

	return batchData, batchLabel

def defNet(norm, layers, loss, drop, activ, optimizer, learningRate):
	l = copy.deepcopy(layers)
	x = tf.placeholder(tf.float32, shape=[None, 310])
	y = tf.placeholder(tf.int32, shape=[None])
	if(norm==1):
		print('Batch Norm')
		mean, varience = tf.nn.moments(x, 0)
		n = tf.nn.batch_normalization(x, mean, varience, None, None, 1e-8)
	elif(norm==2):
		print('L2 Norm')
		n = tf.nn.l2_normalize(x, 0)
	else:
		n = x
	l.insert(0, 310)
	a = n
	for i in range(len(l)-1):
		a = tf.add(tf.matmul(a, tf.Variable(tf.truncated_normal([l[i], l[i+1]], stddev=0.1))), tf.Variable(tf.zeros(l[i+1])))
		if(1==int(drop)):
			a = tf.nn.dropout(a, drop)
		if(activ):
			a = tf.nn.sigmoid(a)
		else:
			a = tf.nn.relu(a)
	a = tf.add(tf.matmul(a, tf.Variable(tf.truncated_normal([l[-1], 3], stddev=0.1))), tf.Variable(tf.zeros(3)))
	if(loss):
		loss = tf.nn.l2_loss(tf.nn.softmax(a) - tf.one_hot(y, 3))
	else:
		loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=a))
	if(optimizer==0):
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=learningRate)
	elif(optimizer==1):
		optimizer = tf.train.AdamOptimizer()
	elif(optimizer==2):
		optimizer = tf.train.AdadeltaOptimizer()
	elif(optimizer==3):
		optimizer = tf.train.AdagradOptimizer(learning_rate=learningRate)
	elif(optimizer==4):
		optimizer = tf.train.RMSPropOptimizer(learning_rate=learningRate)
	train = optimizer.minimize(loss)
	accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(a, 1), tf.to_int64(y)), tf.float64))
	#summary = tf.summary.merge_all()
	#summary_writer = tf.summary.FileWriter(logPath, sess.graph)

	return x, y, train, loss, accuracy

def main():
	args = parseArg()

	dataPath = '../Data/EEG/EEG_X.mat'
	labelPath = '../Data/EEG/EEG_Y.mat'
	logPath = '../Log/EEG'
	data, label = readData(dataPath, labelPath)

	sess = tf.Session()
	testData = data[(args.cross_valid*3394):((args.cross_valid+1)*3394-1)][:]
	testLabel = label[(args.cross_valid*3394):((args.cross_valid+1)*3394-1)][:]
	trainData = np.delete(data, range(args.cross_valid*3394,(args.cross_valid+1)*3394-1), axis=0)
	trainLabel = np.delete(label, range(args.cross_valid*3394,(args.cross_valid+1)*3394-1), axis=0)
	testLabel = np.transpose(testLabel)
	testLabel = testLabel[0]
		
	x, y, train, loss, accuracy = defNet(args.norm, args.layers, args.loss, args.drop_out, args.activ, args.optimizer, args.learning_rate)
	init = tf.global_variables_initializer()
		
	sess.run(init)
	if(args.memory_usage>0):
		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.memory_usage)
		sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) 
	tf.summary.scalar('Accuracy_CV%d'%(args.cross_valid), accuracy)
	summary = tf.summary.merge_all()
	summary_writer = tf.summary.FileWriter(logPath, sess.graph)

	accMax = 0
	print('Training CV %d ...' % args.cross_valid)
	for e in range(args.epoch*int(testData.shape[0]/args.batch_size)):
		batchData, batchLabel = getBatch(trainData, trainLabel, args.batch_size, args.batch_rule)
		batchLabel = np.transpose(batchLabel)
		batchLabel = batchLabel[0]
		_, batchLoss = sess.run([train, loss], feed_dict={x:batchData, y:batchLabel})
		#batchAccuracy = sess.run(accuracy, feed_dict={x:testData, y:testLabel})
		batchAccuracy, batchSummary = sess.run([accuracy, summary], feed_dict={x:testData, y:testLabel})
		summary_writer.add_summary(batchSummary, e)
		if(args.info):
			print('Iter %d Loss %f Acc %f' %(e, batchLoss, batchAccuracy))
		if(batchAccuracy>accMax):
			accMax = batchAccuracy
	print('Max Accuracy %f' % accMax)
	sess.close()

if __name__ == '__main__':
    main()