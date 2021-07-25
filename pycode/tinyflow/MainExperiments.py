from VGG16_test_leo import run_exp as VGG
from Inceptionv3_test_leo import run_exp as InceptionV3
from Inceptionv4_test_leo import run_exp as InceptionV4
from ResNet50_test_leo import run_exp as ResNet50
from DenseNet_test_leo import run_exp as DenseNet
from MultiDynamicWorkloadsExperiments import MDW

InceptionV3([['./log/Inception V3 x1/', 3, 1, 2]])

"""
VGG([['./log/VGG/', 3, 1, 16], ['./log/VGG x1/', 3, 1, 2], ['./log/VGG x2/', 3, 2, 2], ['./log/VGG x3/', 3, 3, 2]])
InceptionV3([['./log/Inception V3/', 3, 1, 16],['./log/Inception V3 x1/', 3, 1, 2],['./log/Inception V3 x2/', 3, 2, 2], ['./log/Inception V3 x3/', 3, 3, 2]])
InceptionV4([['./log/Inception V4/', 3, 1, 16], ['./log/Inception V4 x1/', 3, 1, 2], ['./log/Inception V4 x2/', 3, 2, 2], ['./log/Inception V4 x3/', 3, 3, 2]])
ResNet50([['./log/ResNet/', 3, 1, 16], ['./log/ResNet x1/', 3, 1, 2], ['./log/ResNet x2/', 3, 2, 2], ['./log/ResNet x3/', 3, 3, 2]])
DenseNet([['./log/DenseNet/', 3, 1, 16], ['./log/DenseNet x1/', 3, 1, 2], ['./log/DenseNet x2/', 3, 2, 2], ['./log/DenseNet x3/', 3, 3, 2]])

VGG([['./log/VGG bs4/', 3, 1, 4], ['./log/VGG bs8/', 3, 1, 8], ['./log/VGG bs32/', 3, 1, 32]])
InceptionV3([['./log/Inception V3 bs4/', 3, 1, 4], ['./log/Inception V3 bs8/', 3, 1, 8], ['./log/Inception V3 bs32/', 3, 1, 32]])
InceptionV4([['./log/Inception V4 bs4/', 3, 1, 4], ['./log/Inception V4 bs8/', 3, 1, 8]])
ResNet50([['./log/ResNet bs4/', 3, 1, 4], ['./log/ResNet bs8/', 3, 1, 8], ['./log/ResNet bs32/', 3, 1, 32]])
DenseNet([['./log/DenseNet bs4/', 3, 1, 4], ['./log/DenseNet bs8/', 3, 1, 8]])

MDW()
"""