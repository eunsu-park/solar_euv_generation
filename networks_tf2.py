import tensorflow as tf
import tensorflow.keras as keras


conv_init=tf.random_normal_initializer(0., 0.02)
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

def zero_pad(*a, **k):
    return keras.layers.ZeroPadding2D(*a, **k)

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

def block_encoder(layer, nb_feature):
    layer=leaky_relu(0.2) (layer)
    layer=down_conv(nb_feature, kernel_size=4, strides=2, padding='same') (layer)
    last_=batch_norm() (layer)
    return last_

def block_intmiss(layer, nb_feature):
    layer=leaky_relu(0.2) (layer)
    layer=down_conv(nb_feature, kernel_size=4, strides=2, padding='same') (layer)
    layer=activation('relu') (layer)
    layer=up_conv(nb_feature, kernel_size=4, strides=2, padding='same') (layer)
    layer=batch_norm() (layer)
    last_=dropout(0.5) (layer)
    return last_

def block_decoder(layer, nb_feature):
    layer=activation('relu') (layer)
    layer=up_conv(nb_feature, kernel_size=4, strides=2, padding='same') (layer)
    last_=batch_norm() (layer)
    return last_

def unet_generator(isize, ch_input, ch_output, nb_feature_g=64, nb_feature_max=512, use_tanh=False):

    input_A=keras.layers.Input(shape=(isize, isize, ch_input), dtype=tf.float32)
    current_size=isize
    list_nb_feature=[]
    list_layer_encoder=[]

    nb_feature=min(nb_feature_g, nb_feature_max)
    list_nb_feature.append(nb_feature)
    list_layer_encoder.append(down_conv(nb_feature, kernel_size=4, strides=2, padding='same') (input_A))
    current_size //=2
        
    nb_block=0
    while current_size !=2 :
        
        nb_feature=min(nb_feature*2, nb_feature_max)
        list_nb_feature.append(nb_feature)
        list_layer_encoder.append(block_encoder(list_layer_encoder[-1], nb_feature))
        current_size //=2
        nb_block +=1
            
    layer=block_intmiss(list_layer_encoder[-1], nb_feature)

    list_layer_encoder=list(reversed(list_layer_encoder))
    list_nb_feature=list(reversed(list_nb_feature[:-1]))

    for n in range(nb_block):
        layer=concatenate(axis=-1) ([layer, list_layer_encoder[n]])
        nb_feature=list_nb_feature[n]
        layer=block_decoder(layer, nb_feature)
        current_size *=2
        if current_size <=8 :
            layer=dropout(0.5) (layer)
    
    layer=concatenate (axis=-1)([layer, list_layer_encoder[-1]])
    layer=activation('relu') (layer)
    last_=up_conv(ch_output, kernel_size=4, strides=2, padding='same') (layer)
    current_size *=2
    if use_tanh :
        last_=activation('tanh') (last_)
    
    return keras.models.Model(inputs=input_A, outputs=last_)


def patch_discriminator(isize, ch_input, ch_output, layer_max_d=3, nb_feature_d=64, nb_feature_max=512):

    input_A=keras.layers.Input(shape=(isize, isize, ch_input), dtype=tf.float32, name='input')
    output_B=keras.layers.Input(shape=(isize, isize, ch_output), dtype=tf.float32, name='target')
    nb_feature=nb_feature_d
    
    layer=concatenate(-1)([input_A, output_B])
    
    if layer_max_d==0:
        layer=down_conv(64, kernel_size=1, strides=1, padding='same') (layer)
        layer=leaky_relu(0.2) (layer)
        layer=down_conv(128, kernel_size=1, strides=1, padding='same') (layer)
        layer=batch_norm() (layer)
        layer=leaky_relu(0.2) (layer)
        layer=down_conv(1, kernel_size=1, strides=1, padding='same') (layer)
        last_=activation('sigmoid') (layer)

    else :
        for i in range(layer_max_d):
            layer=down_conv(nb_feature, kernel_size=4, strides=2, padding='same') (layer)
            if i > 0 :
                layer=batch_norm() (layer)
            layer=leaky_relu(0.2) (layer)
            nb_feature=min(nb_feature*2, nb_feature_max)

        layer=zero_pad(1) (layer)
        layer=down_conv(nb_feature, kernel_size=4, strides=1) (layer)
        layer=batch_norm() (layer)
        layer=leaky_relu(0.2) (layer)
        layer=zero_pad(1) (layer)
        layer=down_conv(1, kernel_size=4, strides=1) (layer)
        last_=activation('sigmoid') (layer)

    return keras.models.Model(inputs=[input_A, output_B], outputs=last_)

if __name__ == '__main__' :
    network_D = patch_discriminator(256, 1, 1)
    network_G = unet_generator(256, 1, 1)
    network_D.summary()
    network_G.summary()
