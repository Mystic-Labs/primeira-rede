from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
import tensorflow as tf
sess = tf.InteractiveSession()
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

x = tf.placeholder(tf.float32, shape=[None, 784]) # definicao do input
y_ = tf.placeholder(tf.float32, shape=[None, 10])#deficao de output para sistema com 10 simbolos
W = tf.Variable(tf.zeros([784,10]))#definicao de pesos
b = tf.Variable(tf.zeros([10]))# definicao de bia ou vies sei la
sess.run(tf.global_variables_initializer())
y = tf.matmul(x,W) + b #rede neural em si e so isso mesmo esse modelo
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)#definicao de otimizador ,taxa de aprendizado
for _ in range(1000):#quantidade de treinos
  batch = mnist.train.next_batch(100) # lote de treino
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})#treino em si
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1)) #
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
