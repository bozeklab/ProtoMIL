from model import construct_PPNet
from save import load_model_from_train_state
from settings import get_settings
from train_and_test import test
from datasets import get_test
import numpy as np

config = get_settings('mito')


ppnet_test = construct_PPNet(base_architecture=config.base_architecture,
                             pretrained=config.pretrained,
                             img_size=config.img_size,
                             prototype_shape=config.prototype_shape,
                             num_classes=config.num_classes,
                             prototype_activation_function=config.prototype_activation_function,
                             add_on_layers_type=config.add_on_layers_type,
                             batch_norm_features=config.batch_norm_features)

path_to_model_with_max_push_acc = 'saved_models/mito.ada04.2024-05-13T12:36:48/end.141.100.00.pck'
print('load model from ' + path_to_model_with_max_push_acc)
load_model_from_train_state(path_to_model_with_max_push_acc, ppnet_test)
ppnet_test = ppnet_test.cuda()

accu = []
for i in range(20):
    # TODO dawanie innego seeda jest debilne bo mieszam set treningowy do testowego xddd
    accu.append(test(model=ppnet_test, dataloader=get_test('mito', seed=i, config=config)[1], config=config))

accu = np.array(accu)
print(f'{np.mean(accu)=} +/- {np.std(accu)=}')
