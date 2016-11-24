import numpy as np
import time


np.random.seed(13)
mDim=2000
w1=np.zeros([mDim*400],dtype='float')
b=np.zeros([mDim],dtype='float')
w1=np.random.normal(loc=0.0, scale=1.0, size=400*mDim)
b=np.random.uniform(0,2*np.pi,size=mDim)
gamma = 1
epsilon = 0.000001
lam = 1
def transform(x_original):
    global w1
    global b
    global gamma
    global mDim
    start = time.clock()
    w1_T=np.transpose(w1)
    Xtrans=np.dot(gamma*x_original,np.reshape(w1, (400, mDim)))
    Xtrans = np.concatenate((np.cos(Xtrans+np.reshape(b, (1, mDim))),np.cos(-Xtrans+np.reshape(b, (1, mDim)))),axis = 1)
    Xtrans*= np.sqrt(2.) / np.sqrt(mDim)
    print('Time elapsed for mapper: ', time.clock()-start)
    return Xtrans

def mapper(key, value):
    global lam
    global epsilon
    start = time.clock()
    # key: None
    # value: one line of input file
    alpha = 0.003
    beta_1 = 0.9
    beta_2 = 0.99
    weights = np.zeros([mDim], dtype='float') #maybe adjust this
    m = np.zeros([mDim], dtype='float')
    v = np.zeros([mDim], dtype='float')
    m_hat = np.zeros([mDim], dtype='float')
    v_hat = np.zeros([mDim], dtype='float')
    counter = 0.0
    print 'Mapping...'

    values = []
    for i in value:
     	tokens = i.strip()
     	features = np.fromstring(tokens[0:], dtype=float, sep=' ')
	if len(values) == 0:
	    values = [features]
	else:
	    values = np.vstack((values, features))
    #print values.shape
    print 'Permuting...'
    #values = np.random.permutation(values)
    
    y = values[:, 0]
    kaki = values[:, 1:]
         	
    #print labels[0:10]
    #print features[0, 0:10]
    print 'Transforming...'
    x = transform(kaki)
    #print x[0, 0:10]
    randindex = np.random.permutation(np.shape(x)[0])
    for i in range(k):
        randindex = np.concatenate((randindex,np.random.permutation(np.shape(x)[0]),axis=0)
    print 'Fitting...'
    for j in xrange(0, 75):
		for i in xrange(x.shape[0]):
		    counter = counter + 1.0
		    L = np.dot(weights,x[counter-1])
		    if (y[i]*L) <=0:
		        m = beta_1 * m - (1.0 - beta_1)*y[i]*x[counter-1]
		        v = beta_2 * v + (1.0 - beta_2)*(np.multiply(x[counter-1],x[counter-1]))
                    elif (y[i]*L > 0) and (y[i]*L < 1):
                        m = beta_1 * m - (1.0 - beta_1)*y[i]*(1-L)*x[counter-1]
		        v = beta_2 * v + (1.0 - beta_2)*(1-L)*(1-L)*(np.multiply(x[counter-1],x[i]))
		    else:
		    	m = beta_1 * m
		    	v = beta_2 * v
		    m_hat = m / (1.0 - beta_1**(lam*counter))
		    v_hat = v / (1.0 - beta_2**(lam*counter))
		    weights = weights - np.divide((alpha*m_hat), (np.sqrt(v_hat) + epsilon))

    print 'Time elapsed for mapper: ', time.clock()-start
    yield 1,weights


def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    
    weights_final = 0.0
    num_weights = 0.0
    for i in values:
        weights_final = weights_final + i
        num_weights = num_weights + 1.0
    #print num_weights
    yield weights_final/num_weights
