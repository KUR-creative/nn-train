'''
loss
'''
import tensorflow as tf

def jaccard_distance(n_classes, weights=None, smooth=1.):
    ''' 
    Cacluate class by class intersection & union.
    And then calculate smoothed jaccard_coefficient.
    Finally, calculate the jaccard_distance.
    '''
    import numbers
    assert isinstance(smooth, numbers.Number)

    if weights is None: 
        weights = tf.ones( (1,1,n_classes) )
    else:
        assert len(weights) == n_classes
        weights = tf.constant(weights, dtype=tf.float32)

    axis = tuple(range(n_classes))
    def jacc_dist(y_true, y_pred):
        y_true = y_true * weights
        y_pred = y_pred * weights

        intersection = y_pred * y_true
        sum_ = y_pred + y_true
        numerator = tf.math.reduce_sum(intersection, axis) + smooth
        denominator = tf.math.reduce_sum(sum_ - intersection, axis) + smooth
        jacc = tf.math.reduce_mean(numerator / denominator)
        return 1 - jacc
    return jacc_dist

def goto0test_loss(y_true, y_pred):
    return tf.math.abs(tf.math.reduce_sum(y_pred))
