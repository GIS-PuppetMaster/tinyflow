from VGG16_test_leo import run_exp as VGG
from Inceptionv3_test_leo import run_exp as InceptionV3
from Inceptionv4_test_leo import run_exp as InceptionV4
from ResNet50_test_leo import run_exp as ResNet50
from DenseNet_test_leo import run_exp as DenseNet
# InceptionV4([['./log/Inception V4 bs32/', 3, 1, 32]])
# DenseNet([['./log/DenseNet bs32/', 3, 1, 32]])
VGG([['./log/VGG bs4/', 3, 1, 4], ['./log/VGG bs8/', 3, 1, 8], ['./log/VGG bs32/', 3, 1, 32]])
InceptionV3([['./log/Inception V3 bs4/', 3, 1, 4], ['./log/Inception V3 bs8/', 3, 1, 8], ['./log/Inception V3 bs32/', 3, 1, 32]])
InceptionV4([['./log/Inception V4 bs4/', 3, 1, 4], ['./log/Inception V4 bs8/', 3, 1, 8]])
ResNet50([['./log/ResNet bs4/', 3, 1, 4], ['./log/ResNet bs8/', 3, 1, 8], ['./log/ResNet bs32/', 3, 1, 32]])
DenseNet([['./log/DenseNet bs4/', 3, 1, 4], ['./log/DenseNet bs8/', 3, 1, 8]])