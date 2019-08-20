import tensorflow as tf
import tensorflow.keras as keras

#conv_init=keras.initializers.RandomNormal(0., 0.02)
conv_init=tf.random_normal_initializer(0., 0.02)
#gamma_init=keras.initializers.RandomNormal(1., 0.02)
gamma_init=tf.random_normal_initializer(1., 0.02)

def down_conv(nb_feature, *a, **k):
    return keras.layers.Conv2D(filters=nb_feature, *a, **k,
                                kernel_initializer=conv_init, use_bias=False)

def up_conv(nb_feature, *a, **k):
    return keras.layers.Conv2DTranspose(filters=nb_feature, *a, **k,
                                        kernel_initializer=conv_init, use_bias=False)

def batch_norm():
    return keras.layers.BatchNormalization(momentum=0.9, axis=-1, epsilon=1.01e-5,
                                            gamma_initializer=gamma_init)

def up_sample(*a, **k):
    return keras.layers.UpSampling2D(*a, **k)

def dropout(*a, **k):
    return keras.layers.Dropout(*a, **k)

def leaky_relu(*a, **k):
    return keras.layers.LeakyReLU(*a, **k)

def activation(*a, **k):
    return keras.layers.Activation(*a, **k)

def concatenate(*a, **k):
    return keras.layers.Concatenate(*a, **k)

def dense(*a, **k):
    return keras.layers.Dense(*a, **k)

def block_encoder(nb_feature):
    block=keras.Sequential()
    block.add(leaky_relu(0.2))
    block.add(down_conv(nb_feature, kernel_size=4, strides=2, padding='same'))
    block.add(batch_norm())
    return block

def block_intmiss(nb_feature):
    block=keras.Sequential()
    block.add(leaky_relu(0.2))
    block.add(down_conv(nb_feature, kernel_size=4, strides=2, padding='same'))
    block.add(activation('relu'))
    block.add(up_conv(nb_feature, kernel_size=4, strides=2, padding='same'))
    block.add(batch_norm())
    block.add(dropout(0.5))
    return block

def block_decoder(nb_feature):
    block=keras.Sequential()
    block.add(activation('relu'))
    block.add(up_conv(nb_feature, kernel_size=4, strides=2, padding='same'))
    block.add(batch_norm())
    return block

def unet_generator(isize, ch_input, ch_output, nb_feature_g=64, nb_feature_max=512):

    input_A=keras.layers.Input(shape=(isize, isize, ch_input), dtype=tf.float32)
    current_size=isize
    list_nb_feature=[]
    list_layer_encoder=[]

    nb_feature=min(nb_feature_g, nb_feature_max)
    list_nb_feature.append(nb_feature)
    list_layer_encoder.append(down_conv(nb_feature, kernel_size=4, strides=2, padding='same') (input_A))
    current_size //= 2
        
    nb_block=0
    while current_size != 2 :
        
        nb_feature=min(nb_feature*2, nb_feature_max)
        list_nb_feature.append(nb_feature)
        list_layer_encoder.append(block_encoder(nb_feature)(list_layer_encoder[-1]))
        current_size //= 2
        nb_block += 1
            
    layer=block_intmiss(nb_feature) (list_layer_encoder[-1])

    list_layer_encoder=list(reversed(list_layer_encoder))
    list_nb_feature=list(reversed(list_nb_feature[:-1]))

    for n in range(nb_block):
        layer=concatenate(axis=-1) ([layer, list_layer_encoder[n]])
        nb_feature=list_nb_feature[n]
        layer=block_decoder(nb_feature) (layer)
        current_size *= 2
        if current_size <= 8 :
            layer=dropout(0.5) (layer, training=1)
    
    layer=concatenate (axis=-1)([layer, list_layer_encoder[-1]])
    layer=activation('relu') (layer)
    layer=up_conv(ch_output, kernel_size=4, strides=2, padding='same') (layer)
    current_size *= 2
    layer=activation('tanh') (layer)
    
    return keras.models.Model(inputs=input_A, outputs=layer)


def patch_discriminator(isize, ch_input, ch_output, layer_max_d=3, nb_feature_d=64, nb_feature_max=512):

    input_A=keras.layers.Input(shape=(isize, isize, ch_input), dtype=tf.float32)
    input_B=keras.layers.Input(shape=(isize, isize, ch_output), dtype=tf.float32)
    nb_feature=nb_feature_d
    
    input_AB=concatenate(-1)([input_A, input_B])
    block=keras.Sequential()    

    if layer_max_d == 0:
        block.add(down_conv(64, kernel_size=1, strides=1, padding='same'))
        block.add(leaky_relu(0.2))
        block.add(down_conv(128, kernel_size=1, strides=1, padding='same'))
        block.add(batch_norm())
        block.add(leaky_relu(0.2))
        block.add(down_conv(1, kernel_size=1, strides=1, padding='same'))
        block.add(activation('sigmoid'))

    else :
        for i in range(layer_max_d):
            block.add(down_conv(nb_feature, kernel_size=4, strides=2, padding='same'))
            if i > 0 : block.add(batch_norm())
            block.add(leaky_relu(0.2))
            nb_feature=min(nb_feature*2, nb_feature_max)

        block.add(down_conv(nb_feature, kernel_size=4, strides=1, padding='same'))
        block.add(batch_norm())
        block.add(leaky_relu())

        block.add(down_conv(1, kernel_size=4, strides=1, padding='same'))
        block.add(activation('sigmoid'))

    return keras.models.Model(inputs=[input_A, input_B], outputs=block(input_AB))

if __name__ == '__main__' :
    G=unet_generator(256, 1, 1)
    G.summary()

    D=patch_discriminator(256, 1, 1)
    D.summary()














