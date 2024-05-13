# torch_contour

This library contains 2 pytorch layers for performing the diferentiable operations of :

1. contour to mask
2. contour to distance map. 

It can therefore be used to transform a polygon into a binary mask or distance map in a completely differentiable way.
In particular, it can be used to transform the detection task into a segmentation task.

The predicted polygon should be ordered in counter-clockwise.

This layer takes as input:

a polygon of size 2 x n 

and outputs:

a mask or distance map of size 1xHxW.


##Important: The polygon must have values between 0 and 1.

It is therefore important to apply a sigmoid function before the layer.*.
The two layers have no learnable weight, so all it does is apply a function in a derivative way.


##An example is shown in example.ipnb

