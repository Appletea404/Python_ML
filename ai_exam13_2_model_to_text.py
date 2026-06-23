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

    int_conv2d_1_weights = (f['model_weights']['conv2d_1']['sequential']['conv2d_1']['kernel']
                            [:] * 128).astype(int)
    int_conv2d_1_weights = int_conv2d_1_weights & 0xFF
    print(list(int_conv2d_1_weights))
    print(int_conv2d_1_weights.shape)
    filters = np.transpose(int_conv2d_1_weights, (3, 2, 0, 1))
    print(filters)
    print(filters.shape)

    for fidx,filter in enumerate(filters):
        for chidx,channel in enumerate(filter):
            np.savetxt('./weights/conv2d_1_filter{}_ch{}.txt'.format(fidx, chidx), channel,
                       fmt= '%2x', delimiter=' ')

    int_conv2d_1_bias = (f['model_weights']['conv2d_1']['sequential']['conv2d_1']['bias']
                            [:] * 128).astype(int)
    int_conv2d_1_bias = int_conv2d_1_bias & 0xFF
    print(list(int_conv2d_1_bias))
    print(int_conv2d_1_bias.shape)
    np.savetxt('./weights/conv2d_1_bias.txt', int_conv2d_1_bias,
               fmt='%2x', delimiter=' ')

    int_Dense_weights = (f['model_weights']['dense']['sequential']['dense']['kernel'][:] * 128).astype(int)
    int_Dense_weights = int_Dense_weights & 0xFF
    print(list(int_Dense_weights))
    print(int_Dense_weights.shape)
    np.savetxt('./weights/dense_weights.txt', int_Dense_weights,fmt='%2x', delimiter=' ')

    int_Dense_bias = (f['model_weights']['dense']['sequential']['dense']['bias']
                         [:] * 128).astype(int)
    int_Dense_bias = int_Dense_bias & 0xFF
    print(list(int_Dense_bias))
    print(int_Dense_bias.shape)
    np.savetxt('./weights/dense_bias.txt', int_Dense_bias,
               fmt='%2x', delimiter=' ')

    int_Dense_1_weights = (f['model_weights']['dense_1']['sequential']['dense_1']['kernel'][:] * 128).astype(int)
    int_Dense_1_weights = int_Dense_1_weights & 0xFF
    print(list(int_Dense_1_weights))
    print(int_Dense_1_weights.shape)
    np.savetxt('./weights/dense_1_weights.txt', int_Dense_1_weights, fmt='%2x', delimiter=' ')

    int_Dense_1_bias = (f['model_weights']['dense_1']['sequential']['dense_1']['bias']
                      [:] * 128).astype(int)
    int_Dense_1_bias = int_Dense_1_bias & 0xFF
    print(list(int_Dense_1_bias))
    print(int_Dense_1_bias.shape)
    np.savetxt('./weights/dense_1_bias.txt', int_Dense_1_bias,
               fmt='%2x', delimiter=' ')