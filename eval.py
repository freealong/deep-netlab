import os
import argparse
import torch
import data_loader as module_data
import model.loss as module_loss
import model.metric as module_metric
import model as module_arch
from train import get_instance
from trainer import eval_metrics


def main(config, resume, save=True):
    # setup data_loader instances
    # data_loader = getattr(module_data, config['data_loader']['type'])(
    #     config['data_loader']['args']['data_dir'],
    #     batch_size=512,
    #     shuffle=False,
    #     validation_split=0.0,
    #     training=False,
    #     num_workers=2
    # )
    try:
        data_loader = get_instance(module_data, 'test_data_loader', config)
        print("using test data loader")
    except KeyError:
        print("no test data loader defined")
        return


    # build model architecture
    model = get_instance(module_arch, 'arch', config)
    model.summary()

    # get function handles of loss and metrics
    try:
        loss_fn = getattr(module_loss, config['loss'])
    except AttributeError:
        loss_fn = getattr(model, config['loss'])
    metric_fns = []
    for met in config['metrics']:
        try:
            metric = getattr(module_metric, met)
        except AttributeError:
            metric = getattr(model, met)
        metric_fns.append(metric)

    # load state dict
    checkpoint = torch.load(resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    all_preds, all_gts = [], []

    total_loss = 0.0
    total_metrics = 0.0
    metric_names = None

    with torch.no_grad():
        for i, (data, target) in enumerate(data_loader):
            data = data.to(device)
            if isinstance(target, list):
                target = [x.to(device) for x in target]
            else:
                target = target.to(device)
            output = model(data)

            # computing loss
            loss = loss_fn(output, target)
            total_loss += loss.item()
            # try to post process output to before calculate metrics
            try:
                output = model.postprocess(output)
            except AttributeError:
                pass
            # computing metrics
            metric_names, metrics = eval_metrics(metric_fns, output, target)
            total_metrics += metrics
            # collect pred and gt
            if save:
                all_preds.extend([x.cpu().numpy() for x in list(output)])
                all_gts.extend([x.cpu().numpy() for x in list(target)])
            # print info
            info = "[{}/{} ({:.0f}%)]: Loss: {:.6f} ".format(
                i,
                len(data_loader),
                100.0 * i / len(data_loader),
                loss)
            for name, metric in zip(metric_names, metrics):
                info += "{}: {:.6f} ".format(name, metric)
            print(info)

    log = {'loss': total_loss / len(data_loader)}
    avg_metrics = (metric_names, (total_metrics / len(data_loader)).tolist())
    log.update({name: met for name, met in zip(avg_metrics[0], avg_metrics[1])})
    print(log)

    if save:
        torch.save({'preds': all_preds, 'gts': all_gts}, "preds_gts.pth")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')

    parser.add_argument('-r', '--resume', default=None, type=str, required=True,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')
    parser.add_argument('-b', '--batch_size', default=None, type=int,
                        help='set batch size of data loader')
    parser.add_argument('-s', '--save', default=True, type=bool,
                        help='save output and ground truth')

    args = parser.parse_args()

    config = torch.load(args.resume)['config']
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"]=args.device
    if args.batch_size:
        config['test_data_loader']['args']['batch_size'] = args.batch_size

    main(config, args.resume, args.save)
