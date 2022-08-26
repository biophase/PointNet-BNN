import os
os.environ['TFF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
from glob import glob
import time
from datetime import timezone, datetime
import tensorflow as tf
import h5py
import numpy as np

########################

from model import get_model_segmentation, get_model_segmentation_bayesian
from losses import *
# from dataset_utils import tf_parse_filename, train_val_split
from data_loader import *

########################

def get_timestamp():
    timestamp = str(datetime.now(timezone.utc))[:16]
    timestamp = timestamp.replace('-', '')
    timestamp = timestamp.replace(' ', '_')
    timestamp = timestamp.replace(':', '')
    return timestamp

tf.random.set_seed(0)


# CLI
PARSER = argparse.ArgumentParser(description='CLI for training pipeline')
PARSER.add_argument('--batch_size', type=int, default=12, help='Batch size per step')
PARSER.add_argument('--epochs', type=int, default=60, help='Number of epochs')
PARSER.add_argument('--learning_rate', type=float, default=1e-3, help='Initial learning rate')

ARGS = PARSER.parse_args()

BATCH_SIZE = ARGS.batch_size
EPOCHS = ARGS.epochs
LEARNING_RATE = ARGS.learning_rate
LR_DECAY_STEPS = 5000
LR_DECAY_RATE = 0.9

INIT_TIMESTAMP = get_timestamp()
NUM_CATEGORIES = 12
NUM_FEATS = 12
DATASET_NAME = 'david_grid_ss_BS-20-00'

RUNNING_BAYESIAN = True



AUTOTUNE = tf.data.experimental.AUTOTUNE


train_generator = Generator_h5 (f'data_prep/INPUT/2_h5/{DATASET_NAME}/train.h5')
val_generator = Generator_h5 (f'data_prep/INPUT/2_h5/{DATASET_NAME}/val.h5')


def get_train_generator(reshuffle = True):
    train_ds = tf.data.Dataset.from_generator (train_generator,\
        output_signature=(
            tf.TensorSpec(shape=(4096,NUM_FEATS),dtype=tf.float32),
            tf.TensorSpec(shape=(4096,NUM_CATEGORIES),dtype=tf.float32)
        )
    )
    if reshuffle : train_ds = train_ds.shuffle(500,reshuffle_each_iteration=True)
    train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=True)
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    return train_ds


if RUNNING_BAYESIAN:
  '''Estimate DS length for bayesian model'''
  print('Estimating dataset length for Bayesian model...')
  ds_time=time.time()
  n_batches = (len(list(get_train_generator(reshuffle = False))))
  print('Found {} batches in training DS'.format(n_batches))
  print('Took: {}'.format(time.time()-ds_time))



# Initialise generators
train_ds = get_train_generator()
val_ds = tf.data.Dataset.from_generator (val_generator,\
    output_signature=(
        tf.TensorSpec(shape=(4096,NUM_FEATS),dtype=tf.float32),
        tf.TensorSpec(shape=(4096,NUM_CATEGORIES),dtype=tf.float32)
    )
)
val_ds = val_ds.batch(BATCH_SIZE, drop_remainder=True)
print('Done!')


# Create model
def get_bn_momentum(step):
    return min(0.99, 0.5 + 0.0002*step)
bn_momentum = tf.Variable(get_bn_momentum(0), trainable=False)
if RUNNING_BAYESIAN:
    model = get_model_segmentation_bayesian (bn_momentum=bn_momentum, n_classes=NUM_CATEGORIES, n_feats=NUM_FEATS)
else:
    model = get_model_segmentation(bn_momentum=bn_momentum, n_classes=NUM_CATEGORIES, n_feats=NUM_FEATS)
model.summary()



# Instantiate optimizer and loss function
def get_lr(initial_learning_rate, decay_steps, decay_rate, step, staircase=False, warm_up=True):
    if warm_up:
        coeff1 = min(1.0, step/2000)
    else:
        coeff1 = 1.0

    if staircase:
        coeff2 = decay_rate ** (step // decay_steps)
    else:
        coeff2 = decay_rate ** (step / decay_steps)

    current = initial_learning_rate * coeff1 * coeff2
    return current
LR_ARGS = {'initial_learning_rate': LEARNING_RATE, 'decay_steps': LR_DECAY_STEPS,
           'decay_rate': LR_DECAY_RATE, 'staircase': False, 'warm_up': True}
lr = tf.Variable(get_lr(**LR_ARGS, step=0), trainable=False)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
loss_fxn = ce_loss


# Instantiate metric objects
train_acc = tf.keras.metrics.CategoricalAccuracy()
train_prec = tf.keras.metrics.Precision()
train_recall = tf.keras.metrics.Recall()
train_miou= tf.keras.metrics.MeanIoU(NUM_CATEGORIES)
val_acc = tf.keras.metrics.CategoricalAccuracy()
val_prec = tf.keras.metrics.Precision()
val_recall = tf.keras.metrics.Recall()
val_miou= tf.keras.metrics.MeanIoU(NUM_CATEGORIES)




# Training
print('Training...')
# print('Steps per epoch =', len(TRAIN_FILES) // BATCH_SIZE)
# print('Total steps =', (len(TRAIN_FILES) // BATCH_SIZE) * EPOCHS)


@tf.function
def train_step(inputs, labels):
    
    include_T = True
    
    # Forward pass with gradient tape and loss calc
    with tf.GradientTape() as tape:
        pred_probs = model(inputs, training=True)
        
        # print ("{}Number of loss terms in model: {}{}".format(bcolors.WARNING,len(model.losses),bcolors.ENDC))

        # add and downscale the KL term when running the bayesian model
        if RUNNING_BAYESIAN and include_T:
            t_net_loss_reg_loss = model.losses[0] # tnet reg at  index 0
            kl = tf.reduce_mean(model.losses[1:]) # kl loss at index 1
            loss = loss_fxn(labels, pred_probs) + t_net_loss_reg_loss + (kl / (4096*n_batches)) #loss = nll + tnet_reg + kl
            
        elif RUNNING_BAYESIAN and not include_T:
            kl = tf.reduce_sum(model.losses) # kl loss at index 0
            loss = loss_fxn(labels, pred_probs)  + (kl / (4096*n_batches)) #loss = nll + tnet_reg + kl
            
        # else find loss for point estimate model
        else:
          if include_T: loss = loss_fxn(labels, pred_probs) + sum(model.losses) # loss with tnet reg
          else: loss = loss_fxn(labels, pred_probs) # loss without Tnet reg


    # Obtain gradients of trainable vars w.r.t. loss and perform update
    gradients = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))


    if RUNNING_BAYESIAN : return pred_probs, loss, kl
    # elif RUNNING_BAYESIAN and not include_T: pred_probs, loss, kl
    else : return pred_probs, loss, 0

@tf.function
def val_step(inputs):
    pred_probs = model(inputs, training=False)
    return pred_probs


train_loss_container = []
train_kl_container = []
train_acc_container = []
train_miou_container = []
val_acc_container = []
val_miou_container = []
time_container = []

step = 0
for epoch in range(EPOCHS):
    epoch_start = time.time()
    print('\nEpoch', epoch)

    # Reset metrics
    train_acc.reset_states()
    train_prec.reset_states()
    train_recall.reset_states()
    train_miou.reset_states()

    val_acc.reset_states()
    val_prec.reset_states()
    val_recall.reset_states()
    val_miou.reset_states()
    

    epoch_loss = None
    # Train on batches
    current_batch_n = 0
    
    for x_train, y_train in train_ds:
        
        tic = time.time()

        ### Feature vector modes: [choose 1 and comment out the other 2]
        # x_train = x_train[...,:9] # xyz, rgb, xyz(norm)
        # x_train = np.concatenate((x_train[...,:3], x_train[...,6:9]), axis=-1) # xyz, xyz(norm)
        # x_train = x_train[...,:3] # xyz only


        train_probs, train_loss, kl_div = train_step(x_train, y_train)
        # if RUNNING_BAYESIAN: print('KL div at step {} is : {}'.format(step, kl_div.numpy()/(4096*n_batches)))
        train_acc.update_state(y_train, train_probs)
        
        y_train_indx = tf.math.argmax(y_train,axis = -1)
        train_probs_indx = tf.math.argmax(train_probs,axis = -1)
        train_miou.update_state(y_train_indx, train_probs_indx) 



        max_idxs = tf.math.argmax(train_probs, axis=1)
        train_one_hot = tf.one_hot(max_idxs, depth=NUM_CATEGORIES, dtype=tf.float32)

   
        step += 1
        bn_momentum.assign(get_bn_momentum(step))
        lr.assign(get_lr(**LR_ARGS, step=step))
        
        epoch_loss = tf.reduce_mean(train_loss).numpy()


    
    # Run validation at the end of epoch

    for x_val, y_val in val_ds:
        

        ### Feature vector modes: 
        # x_val = x_val[...,:9] # xyz, rgb, xyz(norm)
        # x_val = np.concatenate((x_val[...,:3], x_val[...,6:9]), axis=-1) # xyz, xyz(norm)
        # x_train = x_train[...,:3] # xyz only

        val_probs = val_step(x_val)

        y_val_indx = tf.math.argmax(y_val,axis = -1)
        val_probs_indx = tf.math.argmax(val_probs,axis = -1)
        val_miou.update_state(y_val_indx, val_probs_indx) 
        val_acc.update_state(y_val,val_probs)

        max_idxs = tf.math.argmax(val_probs, axis=1)
        val_one_hot = tf.one_hot(max_idxs, depth=NUM_CATEGORIES, dtype=tf.float32)


    # Save every epoch (.save_weights() since bn_momentum instance isn't serializable)
    
    print('model.save_weights() at step', step)
    model.save_weights('model/checkpoints/' + INIT_TIMESTAMP + '/iter-' + str(step), save_format='tf')
    
    epoch_train_acc = train_acc.result().numpy()
    epoch_val_acc = val_acc.result().numpy()
    
    print('time_per_epoch: ', time.time()-epoch_start)
    print('loss: ',epoch_loss)
    if RUNNING_BAYESIAN: print('kl divergence: ', kl_div.numpy()/4096/n_batches)
    print('train_accuracy: ', epoch_train_acc)
    print('train_miou: ', train_miou.result().numpy())
    print('val_accuracy: ', epoch_val_acc)
    print('val_miou: ', val_miou.result().numpy())

    
    train_loss_container.append(epoch_loss)
    train_acc_container.append(epoch_train_acc)
    train_miou_container.append(train_miou.result().numpy())
    if RUNNING_BAYESIAN: train_kl_container.append(kl_div)
    val_acc_container.append(epoch_val_acc)
    val_miou_container.append(val_miou.result().numpy())
    
    time_container.append(time.time()-epoch_start)
    
    
  

print('Done training!')

# print(train_miou_container)

np.savetxt('model/checkpoints/' + INIT_TIMESTAMP + '/train_loss_container.txt', train_loss_container )
np.savetxt('model/checkpoints/' + INIT_TIMESTAMP + '/train_acc_container.txt', train_acc_container )
np.savetxt('model/checkpoints/' + INIT_TIMESTAMP + '/train_miou_container.txt', train_miou_container )
np.savetxt('model/checkpoints/' + INIT_TIMESTAMP + '/val_acc_container.txt', val_acc_container )
np.savetxt('model/checkpoints/' + INIT_TIMESTAMP + '/val_miou_container.txt', val_miou_container )
np.savetxt('model/checkpoints/' + INIT_TIMESTAMP + '/time_container.txt', time_container )
if RUNNING_BAYESIAN: np.savetxt('model/checkpoints/' + INIT_TIMESTAMP + '/train_kl_container.txt', train_kl_container)

9
