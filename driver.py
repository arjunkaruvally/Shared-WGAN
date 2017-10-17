import tensorflow as tf

seed = 55
np.random.seed(seed)
tf.set_random_seed(seed)

class GAN_Network():
  class ModelDistribution(object):
    def __init__(self, mean=4, stdev=0.5):
      self.mead = mean
      self.stdev = stdev


    def sample(self, N):
      samples = np.random.normal(self.mu, self.sigma, N)
      # samples.sort()
      return samples


  class GeneratorNoise(object):
    def __init__(self, range):
      self.range = range

    def sample(self, N):
      return np.linspace(-self.range, self.range, N) + \
          np.random.random(N) * 0.01

  def neural_network(input, hidden1_units):
    with tf.name_scope('hidden1'):
      