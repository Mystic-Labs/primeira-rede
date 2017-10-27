
import numpy as np
import pandas as pd

import tensorflow as tf

def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


dataTR = pd.read_csv('input/train.csv')

images = dataTR.iloc[:,1:].values
images = images.astype(np.float)
labels_flat = dataTR.iloc[:,0].values
labels = dense_to_one_hot(labels_flat, 10)
labels = labels.astype(np.uint8)

dataTE = pd.read_csv('input/test.csv')
images2 = dataTE.iloc[:,0:].values
images2 = images2.astype(np.float)
images2 = np.reshape(images2,[28000,28,28,1])

sess = tf.InteractiveSession()
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


index =0   
def lote(batch):
  global index
  imas=images[index:index+batch] #separa os pixels da imagem
  hotVect= labels[index:index+batch]
  index=index+batch
  return np.reshape(imas,[batch,28,28,1]),hotVect  

y_ = tf.placeholder(tf.float32, shape=[None, 10])#deficao de output para sistema com 10 simbolos


##
# DEFININDO A ENTRADA, DOMÍNIO E DIMENSÃO
##
x = tf.placeholder(tf.float32, shape=[None,28,28,1]) # definicao do input

W = tf.Variable(tf.zeros([200,10]))

##
# LAYERS DA CONVOLUÇÃO
##

#DEFININDO A QUANTIDADE DE CANAIS DE CADA CAMADA DA CONVULUÇÃO
canal1 = 4
canal2 = 8
canal3 = 12

#DEFININDO PESOS DA PRIMEIRA CAMADA.
#sendo 5 e 5 o tamanho do filtro, 1 o tamanho do canal da entrada  e o canal1 definido anteriormente
W1 = tf.Variable(tf.truncated_normal([5,5,1,canal1], stddev=0.1))#definicao de pesos
#DEFININDO VIÉS DA PRIMEIRA CAMADA
B1 = tf.Variable(tf.ones([canal1])/10)# definicao de bia ou vies sei la

#DEFININDO PESOS DA SEGUNDA CAMADA
W2 = tf.Variable(tf.truncated_normal([5,5,canal1,canal2],stddev=0.1))
#DEFININDO O VIES DA SEGUNDA CAMADA
B2 = tf.Variable(tf.ones([canal2])/10)

#DEFININDO OS PESOS DA TERCEIRA
W3 = tf.Variable(tf.truncated_normal([4,4,canal2,canal3],stddev=0.1))
#DEFININDO O VIES DA TERCEIRA CAMADA
B3 = tf.Variable(tf.ones([canal3])/10)


#DEFININDO OS PARAMETROS DA FULLY CONECTED.
Penultima = 200

W4 = tf.Variable(tf.truncated_normal([7*7*canal3, Penultima],stddev=0.1))
B4 = tf.Variable(tf.ones([Penultima])/10)

#DEFININDO OS PARAMETROS DA ULTIMA CAMADA, A CAMADA DE RESPOSTA
W5 = tf.Variable(tf.truncated_normal([Penultima,10],stddev=0.1))#esse é o vetor de respostas 
B5 = tf.Variable(tf.zeros([10])/10) #Bias dos neuronios resposta

##MATRIZ 
Y1 = tf.nn.relu(tf.nn.conv2d(x,W1,strides=[1,1,1,1], padding='SAME')+B1)


Y2 = tf.nn.relu(tf.nn.conv2d(Y1,W2,strides=[1,2,2,1], padding='SAME')+B2)


Y3 = tf.nn.relu(tf.nn.conv2d(Y2,W3,strides=[1,2,2,1], padding='SAME')+B3)


YY = tf.reshape(Y3,shape = [-1,7*7*canal3]) #VETORIZA A MATRIZ PARA A CAMADA TODA CONECTADA (FULLY CONNECTED)

Y4 = tf.nn.relu(tf.matmul(YY, W4) + B4) #Penultima camada fully connected com ativação ReLU 

y = tf.nn.softmax(tf.matmul(Y4,W5)+B5)

#Inicia a seção do tensor flow, com as
sess.run(tf.global_variables_initializer())


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)#definicao de otimizador ,taxa de aprendizado

for _ in range(1000):#quantidade de treinos
  batch =lote(100) # lote de treino/ prepara uma imagem em vetor e um vetor de 10 posicees pra cada caso de treino
  train_step.run(feed_dict={x:batch[0],y_:batch[1]})#treino em si
  


correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1)) #
#mede a ac
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
predict = tf.argmax(y,1)
#print(accuracy.eval(feed_dict={x: images[10001:15555], y_: labels[10001:15555]}))
res=predict.eval(feed_dict={x: images2})
np.savetxt('submission_softmax.csv', 
           np.c_[range(1,len(images2)+1),res], 
           delimiter=',', 
           header = 'ImageId,Label', 
           comments = '', 
           fmt='%d')