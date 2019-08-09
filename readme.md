# Solar UV/EUV Generation

## Network Architectures

Let,

C(f, k, s) denotes as 2D Convolution layer with f filters, filter size of k, stride of s,

CT(f, k, s) as 2D Convolution-Transpose layer with f filters, filter size of k, stride of s,

B as Batch-Normalization layer,

R as ReLU activation layer,

L as Leaky-ReLU activation layer with slope 0.2,

T as Tanh activation layer,

S as Sigmoid activation layer,

and D as Dropout layer with rate 0.5.

### Discriminator Network

We can select the size of the receptive field of the discriminator.
 
 - 1x1 discriminator
C(64,1,1)-L-C(128,1,1)-B-L-C(1,1,1)-S\\

 - 16x16 discriminator
C(64,4,2)-L-C(128,4,1)-B-L-C(1,4,1)-S\\

 - 34x34 discriminator
C(64,4,2)-L-C(128,4,2)-B-L-C(256,4,1)-B-L-C(1,4,1)-S\\

 - 70x70 discriminator
C(64,4,2)-L-C(128,4,2)-B-L-C(256,4,2)-B-L-C(512,4,1)-B-L-C(1,4,1)-S\\

 - 142x142 discriminator
C(64,4,2)-L-C(128,4,2)-B-L-C(256,4,2)-B-L-C(512,4,2)-B-L-C(512,4,1)-B-L-C(1,4,1)-S\\

 - 286x286 discriminator
C(64,4,2)-L-C(128,4,2)-B-L-C(256,4,2)-B-L-C(512,4,2)-B-L-C(512,4,2)-B-L-C(512,4,1)-B-L-C(1,4,1)-S\\

### Generator Network

The generator network is consist of the encoder and the decoder

#### Encoder:

1. C(64,4,2)-L
2. C(128,4,2)-B-L
3. C(256,4,2)-B-L
4. C(512,4,2)-B-L
5. C(512,4,2)-B-L
6. C(512,4,2)-B-L
7. C(512,4,2)-B-L
8. C(512,4,2)-B-L
9. C(512,4,2)-B-L
10. C(512,4,2)-R

#### Decoder:

1. CT(512,4,2)-B-D-R
2. CT(512,4,2)-B-D-R
3. CT(512,4,2)-B-D-R
4. CT(512,4,2)-B-R
5. CT(512,4,2)-B-R
6. CT(512,4,2)-B-R
7. CT(256,4,2)-B-R
8. CT(128,4,2)-B-R
9. CT(64,4,2)-B-R
10. CT(1,4,2)-S

The generator network has skip-connections between $i$-th layers of the encoder and $10-i$-th layers of the decoder like the U-Net architecture.

#### Skip-connection:

- encoder 1st layer - decoder 9th layer
- encoder 2nd layer - decoder 8th layer
- encoder 3rd layer - decoder 7th layer
- encoder 4th layer - decoder 6th layer
- encoder 5th layer - decoder 5th layer
- encoder 6th layer - decoder 4th layer
- encoder 7th layer - decoder 3rd layer
- encoder 8th layer - decoder 2nd layer
- encoder 9th layer - decoder 1st layer
