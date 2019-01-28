import os
import argparse
import torch
import numpy as np
import cv2
import data_loader as module_data
import model.loss as module_loss
import model.metric as module_metric
import model as module_arch
from train import get_instance
from trainer import eval_metrics
from utils.visualization import draw_detections


def main(config, resume):
    try:
        data_loader = get_instance(module_data, 'test_data_loader', config)
        print("using test data loader")
    except KeyError:
        print("no test data loader defined")
        return


    # build model architecture
    model = get_instance(module_arch, 'arch', config)
    model.summary()

    # load state dict
    checkpoint = torch.load(resume)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        data_iter = iter(data_loader)
        draw_truth = False
        key = ord('n')
        while True:
            if key == ord('q'):
                break
            elif key == ord('t'):
                draw_truth = not draw_truth
            elif key == ord('n'):
                data = data_iter.next()
                input, target = data
                input = input.to(device)
                output = model(input)
                try:
                    output = model.postprocess(output)
                except AttributeError:
                    pass
                np_imgs, np_target = data_loader.visualize_transform(data)
            draw_img = np_imgs[0].copy()
            draw_detections(draw_img, output[0].cpu().numpy(), class_names=data_loader.class_names, percent=True)
            if draw_truth:
                draw_detections(draw_img, np_target[0], class_names=data_loader.class_names, percent=True)

            cv2.imshow("img", draw_img)
            key = cv2.waitKey(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='detection demo')

    parser.add_argument('-r', '--resume', default=None, type=str, required=True,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')

    args = parser.parse_args()

    config = torch.load(args.resume)['config']
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"]=args.device
    config['test_data_loader']['args']['batch_size'] = 1
    config['test_data_loader']['args']['num_workers'] = 1

    main(config, args.resume)