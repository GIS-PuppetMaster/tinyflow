from pycode.tinyflow import autodiff as ad
from pycode.tinyflow import gpu_op, train, ndarray
import numpy as np

def identity_block(inputs, kernel_size, in_filter, out_filters, block_name, stride=1):
    f1, f2, f3 = out_filters

    W1 = ad.Variable(block_name + "W1")
    W2 = ad.Variable(block_name + "W2")
    W3 = ad.Variable(block_name + "W3")
    rand = np.random.RandomState(seed=123)
    W1_val = rand.normal(scale=0.1, size=(f1, in_filter, 1, 1))
    W2_val = rand.normal(scale=0.1, size=(f2, f1, kernel_size, kernel_size))
    W3_val = rand.normal(scale=0.1, size=(f3, f2, 1, 1))

    # conv1
    conv1 = ad.convolution_2d_forward_op(inputs, W1, "NCHW", "SAME", stride, stride)
    bn1 = ad.bn_forward_op(conv1, "NCHW", "pre_activation")
    act1 = ad.activation_forward_op(bn1, "NCHW", "relu")

    #conv2
    conv2 = ad.convolution_2d_forward_op(act1, W2, "NCHW", "SAME", stride, stride)
    bn2 = ad.bn_forward_op(conv2, "NCHW", "pre_activation")
    act2 = ad.activation_forward_op(bn2, "NCHW", "relu")

    #conv3
    conv3 = ad.convolution_2d_forward_op(act2, W3, "NCHW", "VALID", stride, stride)
    bn3 = ad.bn_forward_op(conv3, "NCHW", "pre_activation")

    #shortcut
    shortcut = inputs
    add = ad.add_op(bn3, shortcut)
    act4 = ad.activation_forward_op(add, "NCHW", "relu")

    dict = {W1: W1_val, W2: W2_val, W3: W3_val}
    return act4, dict

def convolutional_block(inputs, kernel_size, in_filter, out_filters, block_name, stride=1):
    f1, f2, f3 = out_filters

    W1 = ad.Variable(block_name + "W1")
    W2 = ad.Variable(block_name + "W2")
    W3 = ad.Variable(block_name + "W3")
    W_shortcut = ad.Variable(block_name + "W_shortcut")
    rand = np.random.RandomState(seed=123)
    W1_val = rand.normal(scale=0.1, size=(f1, in_filter, 1, 1))
    W2_val = rand.normal(scale=0.1, size=(f2, f1, kernel_size, kernel_size))
    W3_val = rand.normal(scale=0.1, size=(f3, f2, 1, 1))
    W_shortcut_val = rand.normal(scale=0.1, size=(in_filter, f3, 1, 1))

    # conv1
    conv1 = ad.convolution_2d_forward_op(inputs, W1, "NCHW", "SAME", stride, stride)
    bn1 = ad.bn_forward_op(conv1, "NCHW", "pre_activation")
    act1 = ad.activation_forward_op(bn1, "NCHW", "relu")

    # conv2
    conv2 = ad.convolution_2d_forward_op(act1, W2, "NCHW", "SAME", stride, stride)
    bn2 = ad.bn_forward_op(conv2, "NCHW", "pre_activation")
    act2 = ad.activation_forward_op(bn2, "NCHW", "relu")

    # conv3
    conv3 = ad.convolution_2d_forward_op(act2, W3, "NCHW", "VALID", stride, stride)
    bn3 = ad.bn_forward_op(conv3, "NCHW", "pre_activation")

    #shortcut_path
    conv4 = ad.convolution_2d_forward_op(inputs, W_shortcut, "NCHW", "VALID", stride, stride)
    shortcut = ad.bn_forward_op(conv4, "NCHW", "pre_activation")

    # shortcut
    add = ad.add_op(bn3, shortcut)
    act4 = ad.activation_forward_op(add, "NCHW", "relu")

    dict = {W1: W1_val, W2: W2_val, W3: W3_val, W_shortcut: W_shortcut_val}
    return act4, dict

def ResNet152(inputs, n_class):
    X = ad.Placeholder("X")
    y_ = ad.Placeholder("y_")
    W1 = ad.Variable("W1")
    W6 = ad.Variable("W6")
    b6 = ad.Variable("b6")
    W7 = ad.Variable("W7")
    b7 = ad.Variable("b7")
    keep_prob = ad.Placeholder("keep_prob")

    #conv1
    conv1 = ad.convolution_2d_forward_op(X, W1, "NCHW", "VALID", 2, 2)
    bn1 = ad.bn_forward_op(conv1, "NCHW", "pre_activation")
    act1 = ad.activation_forward_op(bn1, "NCHW", "relu")
    pool1 = ad.pooling_2d_forward_op(act1, "NCHW", "max", 0, 0, 2, 2, 3, 3)

    #conv2_x
    conv2, dict2 = convolutional_block(inputs=pool1, kernel_size=3, in_filter=64, out_filters=[64, 64, 256], block_name="2a", stride=1)
    iden2_1, dict2_1 = identity_block(inputs=conv2, kernel_size=3, in_filter=256, out_filters=[64, 64, 256], block_name="2b", stride=1)
    iden2_2, dict2_2 = identity_block(iden2_1, 3, 256, [64, 64, 256], "2c", 1)

    #conv3_x
    conv3, dict3 = convolutional_block(iden2_2, 3, 256, [128, 128, 512], "3a", 1)
    iden3_1, dict3_1 = identity_block(conv3, 3, 512, [128, 128, 512], "3b", 1)
    iden3_2, dict3_2 = identity_block(iden3_1, 3, 512, [128, 128, 512], "3c", 1)
    iden3_3, dict3_3 = identity_block(iden3_2, 3, 512, [128, 128, 512], "3d", 1)
    iden3_4, dict3_4 = identity_block(iden3_3, 3, 512, [128, 128, 512], "3e", 1)
    iden3_5, dict3_5 = identity_block(iden3_4, 3, 512, [128, 128, 512], "3f", 1)
    iden3_6, dict3_6 = identity_block(iden3_5, 3, 512, [128, 128, 512], "3g", 1)
    iden3_7, dict3_7 = identity_block(iden3_6, 3, 512, [128, 128, 512], "3h", 1)

    #conv4_x
    conv4, dict4 = convolutional_block(iden3_7, 3, 512, [256, 256, 1024], "4a", 1)
    iden4_1, dict4_1 = identity_block(conv4, 3, 1024, [256, 256, 1024], "4b", 1)
    iden4_2, dict4_2 = identity_block(iden4_1, 3, 1024, [256, 256, 1024], "4c", 1)
    iden4_3, dict4_3 = identity_block(iden4_2, 3, 1024, [256, 256, 1024], "4d", 1)
    iden4_4, dict4_4 = identity_block(iden4_3, 3, 1024, [256, 256, 1024], "4e", 1)
    iden4_5, dict4_5 = identity_block(iden4_4, 3, 1024, [256, 256, 1024], "4f", 1)
    iden4_6, dict4_6 = identity_block(iden4_5, 3, 1024, [256, 256, 1024], "4f", 1)
    iden4_7, dict4_7 = identity_block(iden4_6, 3, 1024, [256, 256, 1024], "4f", 1)
    iden4_8, dict4_8 = identity_block(iden4_7, 3, 1024, [256, 256, 1024], "4f", 1)
    iden4_9, dict4_9 = identity_block(iden4_8, 3, 1024, [256, 256, 1024], "4f", 1)
    iden4_10, dict4_10 = identity_block(iden4_9, 3, 1024, [256, 256, 1024], "4f", 1)
    iden4_11, dict4_11 = identity_block(iden4_10, 3, 1024, [256, 256, 1024], "4f", 1)
    iden4_12, dict4_12 = identity_block(iden4_11, 3, 1024, [256, 256, 1024], "4f", 1)
    iden4_13, dict4_13 = identity_block(iden4_12, 3, 1024, [256, 256, 1024], "4f", 1)
    iden4_14, dict4_14 = identity_block(iden4_13, 3, 1024, [256, 256, 1024], "4f", 1)
    iden4_15, dict4_15 = identity_block(iden4_14, 3, 1024, [256, 256, 1024], "4f", 1)
    iden4_16, dict4_16 = identity_block(iden4_15, 3, 1024, [256, 256, 1024], "4f", 1)
    iden4_17, dict4_17 = identity_block(iden4_16, 3, 1024, [256, 256, 1024], "4f", 1)
    iden4_18, dict4_18 = identity_block(iden4_17, 3, 1024, [256, 256, 1024], "4f", 1)
    iden4_19, dict4_19 = identity_block(iden4_18, 3, 1024, [256, 256, 1024], "4f", 1)
    iden4_20, dict4_20 = identity_block(iden4_19, 3, 1024, [256, 256, 1024], "4f", 1)
    iden4_21, dict4_21 = identity_block(iden4_20, 3, 1024, [256, 256, 1024], "4f", 1)
    iden4_22, dict4_22 = identity_block(iden4_21, 3, 1024, [256, 256, 1024], "4f", 1)
    iden4_23, dict4_23 = identity_block(iden4_22, 3, 1024, [256, 256, 1024], "4f", 1)
    iden4_24, dict4_24 = identity_block(iden4_23, 3, 1024, [256, 256, 1024], "4f", 1)
    iden4_25, dict4_25 = identity_block(iden4_24, 3, 1024, [256, 256, 1024], "4f", 1)
    iden4_26, dict4_26 = identity_block(iden4_25, 3, 1024, [256, 256, 1024], "4f", 1)
    iden4_27, dict4_27 = identity_block(iden4_26, 3, 1024, [256, 256, 1024], "4f", 1)
    iden4_28, dict4_28 = identity_block(iden4_27, 3, 1024, [256, 256, 1024], "4f", 1)
    iden4_29, dict4_29 = identity_block(iden4_28, 3, 1024, [256, 256, 1024], "4f", 1)
    iden4_30, dict4_30 = identity_block(iden4_29, 3, 1024, [256, 256, 1024], "4f", 1)
    iden4_31, dict4_31 = identity_block(iden4_30, 3, 1024, [256, 256, 1024], "4f", 1)
    iden4_32, dict4_32 = identity_block(iden4_31, 3, 1024, [256, 256, 1024], "4f", 1)
    iden4_33, dict4_33 = identity_block(iden4_32, 3, 1024, [256, 256, 1024], "4f", 1)
    iden4_34, dict4_34 = identity_block(iden4_33, 3, 1024, [256, 256, 1024], "4f", 1)
    iden4_35, dict4_35 = identity_block(iden4_34, 3, 1024, [256, 256, 1024], "4f", 1)


    #conv5_x
    conv5, dict5 = convolutional_block(iden4_35, 3, 1024, [512, 512, 2048], "5a", 1)
    iden5_1, dict5_1 = identity_block(conv5, 3, 2048, [512, 512, 2048], "5b", 1)
    iden5_2, dict5_2 = identity_block(iden5_1, 3, 2048, [512, 512, 2048], "5c", 1)
    pool5 = ad.pooling_2d_forward_op(iden5_2, "NCHW", "mean", 0, 0, 1, 1, 2, 2)


    pool5_flat = ad.flatten_op(pool5)
    mul6 = ad.matmul_op(pool5_flat, W6)
    add6 = ad.add_op(mul6, b6)
    act6 = ad.fullyactivation_forward_op(add6, "NCHW", "relu")
    drop_out = ad.fullydropout_forward_op(act6, "NCHW", keep_prob)
    mul7 = ad.matmul_op(drop_out, W7)
    add7 = ad.add_op(mul7, b7)
    act7 = ad.fullyactivation_forward_op(add7, "NCHW", "softmax")

    loss = ad.softmaxcrossentropy_op(act7, y_)

    X_val = np.random.normal(0, 0.5, (10, 3, 230, 230))
    W1_val = np.random.normal(0, 0.5, (64, 3, 7, 7))
    W6_val = np.random.normal(0, 0.5, (7*7*2048, 50))
    b6_val = np.random.normal(0, 0.5, (10, 50))
    W7_val = np.random.normal(0, 0.5, (50, 6))
    b7_val = np.random.normal(0, 0.5, (10, 6))
    y_val = np.random.normal(0, 0.5, (10, 6))

    aph = 0.001
    t = train.Adam_minimize(loss, aph)
    feed_dict = {W1: W1_val, W6: W6_val, W7: W7_val, b6: b6_val, b7: b7_val}
    feed_dict.update(dict2)
    feed_dict.update(dict2_1)
    feed_dict.update(dict2_2)
    feed_dict.update(dict3)
    feed_dict.update(dict3_1)
    feed_dict.update(dict3_2)
    feed_dict.update(dict3_3)
    feed_dict.update(dict3_4)
    feed_dict.update(dict3_5)
    feed_dict.update(dict3_6)
    feed_dict.update(dict3_7)
    feed_dict.update(dict4)
    feed_dict.update(dict4_1)
    feed_dict.update(dict4_2)
    feed_dict.update(dict4_3)
    feed_dict.update(dict4_4)
    feed_dict.update(dict4_5)
    feed_dict.update(dict4_6)
    feed_dict.update(dict4_7)
    feed_dict.update(dict4_8)
    feed_dict.update(dict4_9)
    feed_dict.update(dict4_10)
    feed_dict.update(dict4_11)
    feed_dict.update(dict4_12)
    feed_dict.update(dict4_13)
    feed_dict.update(dict4_14)
    feed_dict.update(dict4_15)
    feed_dict.update(dict4_16)
    feed_dict.update(dict4_17)
    feed_dict.update(dict4_18)
    feed_dict.update(dict4_19)
    feed_dict.update(dict4_20)
    feed_dict.update(dict4_21)
    feed_dict.update(dict4_22)
    feed_dict.update(dict4_23)
    feed_dict.update(dict4_24)
    feed_dict.update(dict4_25)
    feed_dict.update(dict4_26)
    feed_dict.update(dict4_27)
    feed_dict.update(dict4_28)
    feed_dict.update(dict4_29)
    feed_dict.update(dict4_30)
    feed_dict.update(dict4_31)
    feed_dict.update(dict4_32)
    feed_dict.update(dict4_33)
    feed_dict.update(dict4_34)
    feed_dict.update(dict4_35)
    feed_dict.update(dict5)
    feed_dict.update(dict5_1)
    feed_dict.update(dict5_2)

    # t.init_Variable(feed_dict)
    # t.run({X: X_val, y_: y_val})
    # print(t.get_Variable_node_to_val_map()[W1_val].asnumpy())
    list = []
    for key in feed_dict.keys():
        list.append(key)
    executor = ad.Executor(list, ctx=ndarray.gpu(0))
    executor.run(feed_dict)


ResNet152(1, 1)






