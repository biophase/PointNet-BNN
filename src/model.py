import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Layer, Conv2D, BatchNormalization, Dense, Dropout
import tensorflow_probability as tfp
from viz_utils import bcolors

###################################################################################################################
#############################  Segmentation using 1024 local pointwise features  ##################################
###################################################################################################################
# 

class TNet(Layer):
    def __init__(self, add_regularization=False, bn_momentum=0.99, **kwargs):
        super(TNet, self).__init__(**kwargs)
        self.add_regularization = add_regularization
        self.bn_momentum = bn_momentum
        self.conv0 = CustomConv(64, (1, 1), strides=(1, 1), bn_momentum=bn_momentum)
        self.conv1 = CustomConv(128, (1, 1), strides=(1, 1), bn_momentum=bn_momentum)
        self.conv2 = CustomConv(1024, (1, 1), strides=(1, 1), bn_momentum=bn_momentum)
        self.fc0 = CustomDense(512, activation=tf.nn.relu, apply_bn=True, bn_momentum=bn_momentum)
        self.fc1 = CustomDense(256, activation=tf.nn.relu, apply_bn=True, bn_momentum=bn_momentum)

    def build(self, input_shape):
        self.K = input_shape[-1]

        self.w = self.add_weight(shape=(256, self.K**2), initializer=tf.zeros_initializer,
                                 trainable=True, name='w')
        self.b = self.add_weight(shape=(self.K, self.K), initializer=tf.zeros_initializer,
                                 trainable=True, name='b')

        # Initialize bias with identity
        I = tf.constant(np.eye(self.K), dtype=tf.float32)
        self.b = tf.math.add(self.b, I)

    def call(self, x, training=None):
        input_x = x                                                     # BxNxK

        # Embed to higher dim
        x = tf.expand_dims(input_x, axis=2)                             # BxNx1xK
        x = self.conv0(x, training=training)
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        x = tf.squeeze(x, axis=2)                                       # BxNx1024

        # Global features
        x = tf.reduce_max(x, axis=1)                                    # Bx1024

        # Fully-connected layers
        x = self.fc0(x, training=training)                              # Bx512
        x = self.fc1(x, training=training)                              # Bx256

        # Convert to KxK matrix to matmul with input
        x = tf.expand_dims(x, axis=1)                                   # Bx1x256
        x = tf.matmul(x, self.w)                                        # Bx1xK^2
        x = tf.squeeze(x, axis=1)
        x = tf.reshape(x, (-1, self.K, self.K))

        # Add bias term (initialized to identity matrix)
        x += self.b

        # Add regularization
        if self.add_regularization:
            eye = tf.constant(np.eye(self.K), dtype=tf.float32)
            x_xT = tf.matmul(x, tf.transpose(x, perm=[0, 2, 1]))
            reg_loss = tf.nn.l2_loss(eye - x_xT)
            self.add_loss(1e-3 * reg_loss)

        return tf.matmul(input_x, x)

    def get_config(self):
        config = super(TNet, self).get_config()
        config.update({
            'add_regularization': self.add_regularization,
            'bn_momentum': self.bn_momentum})

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class CustomConv(Layer):
    def __init__(self, filters, kernel_size, strides, padding='valid', activation=None,
                 apply_bn=False, bn_momentum=0.99, **kwargs):
        super(CustomConv, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.apply_bn = apply_bn
        self.bn_momentum = bn_momentum
        self.conv = Conv2D(filters, kernel_size, strides=strides, padding=padding,
                           activation=activation, use_bias=not apply_bn)
        if apply_bn:
            self.bn = BatchNormalization(momentum=bn_momentum, fused=False)

    def call(self, x, training=None):
        x = self.conv(x)
        if self.apply_bn:
            x = self.bn(x, training=training)
        if self.activation:
            x = self.activation(x)
        return x

    def get_config(self):
        config = super(CustomConv, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'activation': self.activation,
            'apply_bn': self.apply_bn,
            'bn_momentum': self.bn_momentum})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class CustomDense(Layer):
    def __init__(self, units, activation=None, apply_bn=False, bn_momentum=0.99, **kwargs):
        super(CustomDense, self).__init__(**kwargs)
        self.units = units
        self.activation = activation
        self.apply_bn = apply_bn
        self.bn_momentum = bn_momentum
        self.dense = Dense(units, activation=activation, use_bias=not apply_bn)
        if apply_bn:
            self.bn = BatchNormalization(momentum=bn_momentum, fused=False)

    def call(self, x, training=None):
        x = self.dense(x)
        if self.apply_bn:
            x = self.bn(x, training=training)
        if self.activation:
            x = self.activation(x)
        return x

    def get_config(self):
        config = super(CustomDense, self).get_config()
        config.update({
            'units': self.units,
            'activation': self.activation,
            'apply_bn': self.apply_bn,
            'bn_momentum': self.bn_momentum})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)



def get_model_segmentation(bn_momentum, n_classes = 14):
    # TODO: should Input shape be (None,9) or (4096,9) ?
    include_T = True
    pt_cloud = Input(shape=(None, 9), dtype=tf.float32, name='pt_cloud')    # BxNx3
    
    # pt_cloud = tf.concat((pt_cloud[...,:3],pt_cloud[...,3:]/255.), axis = -1)
    # pt_cloud = pt_cloud - tf.repeat(tf.reduce_mean(pt_cloud, axis = 1, keepdims = True), repeats = 4096, axis=1)
    

    batch_size = pt_cloud.shape[0]
    n_points = pt_cloud.shape[1] # should be 4096

    # Input transformer (B x N x 3 -> B x N x 3)
    if include_T: pt_cloud_transform = TNet(bn_momentum=bn_momentum)(pt_cloud)
    else : pt_cloud_transform = pt_cloud

    # Embed to 64-dim space (B x N x 3 -> B x N x 64)
    pt_cloud_transform = tf.expand_dims(pt_cloud_transform, axis=2)         # for weight-sharing of conv
    hidden_64 = CustomConv(64, (1, 1), strides=(1, 1), activation=tf.nn.relu, apply_bn=True,
                           bn_momentum=bn_momentum)(pt_cloud_transform)
    embed_64 = CustomConv(64, (1, 1), strides=(1, 1), activation=tf.nn.relu, apply_bn=True,
                          bn_momentum=bn_momentum)(hidden_64)
    embed_64 = tf.squeeze(embed_64, axis=2)

    # Feature transformer (B x N x 64 -> B x N x 64)
    if include_T: embed_64_transform = TNet(bn_momentum=bn_momentum, add_regularization=True)(embed_64)
    else: embed_64_transform = embed_64
    

    # Embed to 1024-dim space (B x N x 64 -> B x N x 1024)
    embed_64_transform = tf.expand_dims(embed_64_transform, axis=2)
    hidden_64 = CustomConv(64, (1, 1), strides=(1, 1), activation=tf.nn.relu, apply_bn=True,
                           bn_momentum=bn_momentum)(embed_64_transform)
    hidden_128 = CustomConv(128, (1, 1), strides=(1, 1), activation=tf.nn.relu, apply_bn=True,
                            bn_momentum=bn_momentum)(hidden_64)
    embed_1024 = CustomConv(1024, (1, 1), strides=(1, 1), activation=tf.nn.relu, apply_bn=True,
                            bn_momentum=bn_momentum)(hidden_128)
    embed_1024 = tf.squeeze(embed_1024, axis=2)
    passed_local_features = embed_1024

    # Global feature vector (B x N x 1024 -> B x 1024)
    global_descriptor = tf.reduce_max(embed_1024, axis=1)
    
    fc_256 = CustomDense(256, activation=tf.nn.relu, apply_bn=True,
                             bn_momentum=bn_momentum)(global_descriptor)
    fc_128 = CustomDense(128, activation=tf.nn.relu, apply_bn=True,
                             bn_momentum=bn_momentum)(fc_256) 
    #fc_128 = Dropout(rate = 0.3)(fc_128)
    print("global descriptor shape is: ", fc_128.shape)

    # Tile global feature vector (B x 1024 -> B x N x 1024)
    global_descriptor_expanded = tf.expand_dims(fc_128, axis = 1)
    # global_descriptor_expanded = tf.reshape(global_descriptor_expanded,[batch_size,1,-1]) # Bx1x1024
    global_descriptor_expanded_tiled = tf.tile(global_descriptor_expanded,[1,4096,1]) # BxNx1024
    
    ### TODO: why can't I tile a tensor by [1, n_points, 1], where n_points will be known at run time. 
    # Only hard-coded possible?

    
    # Concatenate with point features 
    point_global_descriptors = tf.concat([passed_local_features, 
                        global_descriptor_expanded_tiled], axis = 2) # BxNx1088
    point_global_descriptors = tf.expand_dims(point_global_descriptors, axis=2)

    embed_512 = CustomConv(512, (1, 1), strides=(1, 1), activation=tf.nn.relu, apply_bn=True,
                           bn_momentum=bn_momentum)(point_global_descriptors)
    embed_256 = CustomConv(256, (1, 1), strides=(1, 1), activation=tf.nn.relu, apply_bn=True,
                           bn_momentum=bn_momentum)(embed_512)
    embed_256 = Dropout(rate = 0.3)(embed_256)
    predictions = CustomConv(n_classes, (1, 1), strides=(1, 1), activation=tf.nn.softmax, apply_bn=True,
                           bn_momentum=bn_momentum)(embed_256)
    predictions = tf.squeeze(predictions, axis=2)
    print('prediction shapes = ', predictions.shape)                           

    return Model(inputs = pt_cloud, outputs = predictions)


###################################################################################################################
#######################################  Bayesian implementation  #################################################
###################################################################################################################



def get_bias_prior(dtype, shape,name, trainable, add_variable_fn):
#   print('bias call event shape: ', shape)
#   print('bias call dtype: ', dtype)
#   print('bias call name: ', name)
#   print('bias call trainable: ', trainable)
#   print('bias call add_variable_fn: ', add_variable_fn)

  prior = tfp.distributions.Independent(tfp.distributions.Normal(
                                      loc = tf.zeros(shape, dtype = dtype),
                                      scale = 8.0 * tf.ones(shape, dtype = dtype)),
                                      reinterpreted_batch_ndims = 1)

  return prior

def get_kernel_prior(dtype, shape, name, trainable, add_variable_fn):
#   print('kernel call event shape: ', shape)
#   print('kernel call dtype: ', dtype)
#   print('kernel call name: ', name)
#   print('kernel call trainable: ', trainable)
#   print('kernel call add_variable_fn: ', add_variable_fn)

  prior = tfp.distributions.Independent(tfp.distributions.Normal(
                                      loc = tf.zeros(shape, dtype = dtype),
                                      scale = 4.0 * tf.ones(shape, dtype = dtype)),
                                      reinterpreted_batch_ndims = 4)

  return prior



class BNN_CustomConv(Layer):
    def __init__(self, filters, kernel_size, strides, padding='valid', activation=None,
                 apply_bn=False, bn_momentum=0.99, **kwargs):
        super(BNN_CustomConv, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.apply_bn = apply_bn
        self.bn_momentum = bn_momentum

        # if self.apply_bn:       
        #     bias_prior_fn = None
        #     bias_posterior_fn = None
        #     bias_posterior_tensor_fn = None
        # elif not self.apply_bn: 
        #     # bias_prior_fn = tfp.layers.default_multivariate_normal_fn
        # bias_prior_fn = get_bias_prior
        # bias_posterior_fn = tfp.layers.util.default_mean_field_normal_fn(is_singular=False)
        # bias_posterior_tensor_fn = (lambda d: d.sample())

        self.conv = tfp.layers.Convolution2DFlipout(
                                filters, 
                                kernel_size,
                                strides=strides,
                                padding=padding,
                                activation=activation,


                                kernel_posterior_fn = tfp.layers.util.default_mean_field_normal_fn(is_singular=False),
                                kernel_posterior_tensor_fn=(lambda d: d.sample()),
                                kernel_prior_fn = get_kernel_prior,
                                kernel_divergence_fn = (lambda q, p, _: tfp.distributions.kl_divergence(q, p, allow_nan_stats=True)),

                                bias_posterior_fn = tfp.layers.util.default_mean_field_normal_fn(is_singular=False),
                                bias_posterior_tensor_fn=(lambda d: d.sample()),
                                bias_prior_fn = get_bias_prior,
                                bias_divergence_fn = (lambda q, p, _: tfp.distributions.kl_divergence(q, p, allow_nan_stats=True))
                                )
        if apply_bn:
            self.bn = BatchNormalization(momentum=bn_momentum, fused=False)

    def call(self, x, training=None):
        x = self.conv(x)
        if self.apply_bn:
            x = self.bn(x, training=training)
        if self.activation:
            x = self.activation(x)
        return x

    def get_config(self):
        config = super(CustomConv, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'activation': self.activation,
            'apply_bn': self.apply_bn,
            'bn_momentum': self.bn_momentum})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class BNN_CustomDense(Layer):
    def __init__(self, units, activation=None, apply_bn=False, bn_momentum=0.99, **kwargs):
        super(CustomDense, self).__init__(**kwargs)
        self.units = units
        self.activation = activation
        self.apply_bn = apply_bn
        self.bn_momentum = bn_momentum
        self.dense = Dense(units, activation=activation, use_bias=not apply_bn)
        if apply_bn:
            self.bn = BatchNormalization(momentum=bn_momentum, fused=False)

    def call(self, x, training=None):
        x = self.dense(x)
        if self.apply_bn:
            x = self.bn(x, training=training)
        if self.activation:
            x = self.activation(x)
        return x

    def get_config(self):
        config = super(CustomDense, self).get_config()
        config.update({
            'units': self.units,
            'activation': self.activation,
            'apply_bn': self.apply_bn,
            'bn_momentum': self.bn_momentum})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)



def get_model_segmentation_bayesian(bn_momentum, n_classes, n_feats):
    
    print(bcolors.UNDERLINE + 'Loading bayesian model!' + bcolors.ENDC)
    include_T = True
    pt_cloud = Input(shape=(None, n_feats), dtype=tf.float32, name='pt_cloud')    # BxNx3


    # Input transformer (B x N x 3 -> B x N x 3)
    if include_T: pt_cloud_transform = TNet(bn_momentum=bn_momentum)(pt_cloud)
    else : pt_cloud_transform = pt_cloud

    # Embed to 64-dim space (B x N x 3 -> B x N x 64)
    pt_cloud_transform = tf.expand_dims(pt_cloud_transform, axis=2)         # for weight-sharing of conv
    hidden_64 = CustomConv(64, (1, 1), strides=(1, 1), activation=tf.nn.relu, apply_bn=True,
                            bn_momentum=bn_momentum)(pt_cloud_transform)
    embed_64 = CustomConv(64, (1, 1), strides=(1, 1), activation=tf.nn.relu, apply_bn=True,
                            bn_momentum=bn_momentum)(hidden_64)
    embed_64 = tf.squeeze(embed_64, axis=2)

    # Feature transformer (B x N x 64 -> B x N x 64)
    if include_T: embed_64_transform = TNet(bn_momentum=bn_momentum, add_regularization=True)(embed_64)
    else: embed_64_transform = embed_64


    # Embed to 1024-dim space (B x N x 64 -> B x N x 1024)
    embed_64_transform = tf.expand_dims(embed_64_transform, axis=2)
    hidden_64 = CustomConv(64, (1, 1), strides=(1, 1), activation=tf.nn.relu, apply_bn=True,
                            bn_momentum=bn_momentum)(embed_64_transform)
    hidden_128 = CustomConv(128, (1, 1), strides=(1, 1), activation=tf.nn.relu, apply_bn=True,
                            bn_momentum=bn_momentum)(hidden_64)
    embed_1024 = CustomConv(1024, (1, 1), strides=(1, 1), activation=tf.nn.relu, apply_bn=True,
                            bn_momentum=bn_momentum)(hidden_128)
    embed_1024 = tf.squeeze(embed_1024, axis=2)
    passed_local_features = embed_1024

    # Global feature vector (B x N x 1024 -> B x 1024)
    global_descriptor = tf.reduce_max(embed_1024, axis=1)

    fc_256 = CustomDense(256, activation=tf.nn.relu, apply_bn=True,
                                bn_momentum=bn_momentum)(global_descriptor)
    fc_128 = CustomDense(128, activation=tf.nn.relu, apply_bn=True,
                                bn_momentum=bn_momentum)(fc_256) 
    #fc_128 = Dropout(rate = 0.3)(fc_128)
    print("global descriptor shape is: ", fc_128.shape)

    # Tile global feature vector (B x 1024 -> B x N x 1024)
    global_descriptor_expanded = tf.expand_dims(fc_128, axis = 1)
    # global_descriptor_expanded = tf.reshape(global_descriptor_expanded,[batch_size,1,-1]) # Bx1x1024
    global_descriptor_expanded_tiled = tf.tile(global_descriptor_expanded,[1,4096,1]) # BxNx1024

    ### TODO: why can't I tile a tensor by [1, n_points, 1], where n_points will be known at run time. 
    # Only hard-coded possible?


    # Concatenate with point features 
    point_global_descriptors = tf.concat([passed_local_features, 
                        global_descriptor_expanded_tiled], axis = 2) # BxNx1088
    point_global_descriptors = tf.expand_dims(point_global_descriptors, axis=2)

    embed_512 = BNN_CustomConv(512, (1, 1), strides=(1, 1), activation=tf.nn.relu, apply_bn=True,
                            bn_momentum=bn_momentum)(point_global_descriptors)
    embed_256 = BNN_CustomConv(256, (1, 1), strides=(1, 1), activation=tf.nn.relu, apply_bn=True,
                            bn_momentum=bn_momentum)(embed_512)
    

    predictions = BNN_CustomConv(n_classes, (1, 1), strides=(1, 1), activation=tf.nn.softmax, apply_bn=True,
                            bn_momentum=bn_momentum)(embed_256)

    predictions = tf.squeeze(predictions, axis=2)

    print('prediction shapes : ', predictions.shape)     
    print('prediction type : ', type(predictions))                        

    return Model(inputs = pt_cloud, outputs = predictions)







