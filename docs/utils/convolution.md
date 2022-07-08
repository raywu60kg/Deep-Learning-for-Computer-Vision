# convolution


## calculate convoluation output shape
W = Input size
K = Filter size
S = Stride
P = Padding

output size: 
floor(((W - K + 2P)/S) + 1)

## calculate pooing output shape
W = Input size
F = Filter size
S = stride

output size:
floor((W-F)/S+1)