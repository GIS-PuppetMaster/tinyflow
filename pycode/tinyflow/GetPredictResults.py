import os

from tensorflow.python.eager import executor

os.environ['CUDA_VISIBLE_DEVICES'] = f'{0}'
from VGG16_test_leo import VGG16
from Inceptionv3_test_leo import Inceptionv3
from Inceptionv4_test_leo import Inceptionv4
from ResNet50_test_leo import ResNet50
from DenseNet_test_leo import DenseNet121
import pickle as pkl

def get_predict_results(batch_size, num_step, log_path, job_id, model, **kwargs):
    m = model(num_step=num_step, batch_size=batch_size, log_path=log_path, job_id=job_id)
    return m.get_predict_results(1000)


if __name__ == '__main__':
    if not os.path.exists('./log/temp/schedule/'):
        os.makedirs('./log/temp/schedule/')
    log_path = f'./log/temp/schedule/'
    model_list = [VGG16, Inceptionv3, Inceptionv4, ResNet50, DenseNet121]
    predict_results = [get_predict_results(2, 50, log_path, job_id, model_list[job_id]) for job_id in
                       range(len(model_list))]
    res = {}
    if not os.path.exists('../../res/inferred_shape/'):
        os.makedirs('../../res/inferred_shape/')
    for i, name in enumerate(['VGG16', 'Inceptionv3', 'Inceptionv4', 'ResNet50', 'DenseNet121']):
        res[name] = predict_results[i]
    with open(f'../../res/inferred_shape.pkl', 'wb') as f:
        pkl.dump(res, f)


