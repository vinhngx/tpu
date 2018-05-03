# -*- coding: utf-8 -*-
"""
Created on Thu May  3 11:33:09 2018

@author: vinhn
"""

LABEL_CLASSES = 1000
NUM_TRAIN_IMAGES = 1281167

BATCH_SIZE = 2048
NUM_EPOCHS = 45

IMG_PER_SEC = 3000

print('Num images: %d, batchsize: %d'%(NUM_TRAIN_IMAGES, BATCH_SIZE))
print('Num epochs: %d, Num steps: %d'%(NUM_EPOCHS, NUM_EPOCHS*NUM_TRAIN_IMAGES/BATCH_SIZE))

for epoch in [1, 10, 15, 20, 30, 40, 45]:
    print('Epoch %d steps %d\n'%(epoch, epoch*NUM_TRAIN_IMAGES/BATCH_SIZE))
    
print('Time to %d epochs: %f hours'% (NUM_EPOCHS, NUM_EPOCHS*NUM_TRAIN_IMAGES/IMG_PER_SEC/3600) )    