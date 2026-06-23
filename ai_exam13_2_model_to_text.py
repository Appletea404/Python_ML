import h5py
import numpy as np

filename = './models/cnn_mnist_micro_0.979.h5'
with h5py.File(filename, 'r') as f:
    int_conv2d_weights = (f['model_weights']['conv2d']['sequential']['conv2d']['kernel'][:] * 128).astype(int)

    int_conv2d_weights = int_conv2d_weights & 0xFF
    print(int_conv2d_weights)
    print(int_conv2d_weights.shape)
    filters = int_conv2d_weights[:, :, 0, :]
    print(filters.shape)
    filters = np.transpose(filters, (2, 0, 1))
    print(filters)
    print(filters.shape)
    for i in range(len(filters)):
        np.savetxt('./weights/conv2d_{}.txt'.format(i), filters[i],
        fmt = '%2x', delimiter=' ')

    int_conv2d_bias = (f['model_weights']['conv2d']['sequential']['conv2d']['bias'][:] * 128).astype(int)
    int_conv2d_bias = int_conv2d_bias & 0xFF
    print(int_conv2d_bias)
    np.savetxt('./weights/conv2d_bias.txt',int_conv2d_bias,
               fmt='%2x', delimiter=' ')