a resnet implementation in pytorch.\
\
a resnet works with skip connections, skip connections mean you save the input (and the saved input is called the identity) and after you get the output from the residual block(a residual block is simply a some convolutional layers grouped together) you add the identity to the output.this helps with input signal being lost in deep convolutional layers.\
\
resources to understand res-nets better = https://www.youtube.com/watch?v=o_3mboe1jYI | https://www.youtube.com/watch?v=DkNIBBBvcPs \
you cannot have a input with a dimension lower than 32x32. since the layers half the dimension 5 times.
