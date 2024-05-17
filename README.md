# torch_contour
<figure>
<p align="center">
  <img 
  src="vary_nodes.jpg"
  alt="Example of torch contour on a circle when varying the number of nodes"
  width="500">
  <figcaption>Example of torch contour on a circle when varying the number of nodes</figcaption>
</p>
</figure>

This library contains 2 pytorch layers for performing the diferentiable operations of :

1. contour to mask
2. contour to distance map. 

It can therefore be used to transform a polygon into a binary mask or distance map in a completely differentiable way.
In particular, it can be used to transform the detection task into a segmentation task.
The two layers have no learnable weight, so all it does is apply a function in a derivative way.



## Input (Float):

A polygon of size $2 \times n$ with \
with $n$ the number of nodes


## Output (Float):

A mask or distance map of size $1 \times H \times W$.\
with $H$ and $W$ respectively the Heigh and Width of the distance map or mask.

## Important: 

The polygon must have values between 0 and 1. It is therefore important to apply a sigmoid function before the layer.*.

## Example:

 ```
from torch_contour.torch_contour import Contour_to_distance_map, Contour_to_mask
import torch
import matplotlib.pyplot as plt

x = torch.tensor([[0.1,0.1],
                  [0.1,0.9],
                  [0.9,0.9],
                  [0.9,0.1]])[None]

Dmap = Contour_to_distance_map(200)
Mask = Contour_to_mask(200)

plt.imshow(Dmap(x).cpu().detach().numpy()[0,0])
plt.show()
plt.imshow(Mask(x).cpu().detach().numpy()[0,0])
plt.show()
```

