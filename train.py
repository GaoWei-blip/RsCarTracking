from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import torch.utils.data

from model.DSFNet import DSFNet, load_model_weight, save_model
from common.dataset.coco_rsdata import COCO
from common.utils.logger import Logger
from common.Trainer.ctdet import CtdetTrainer
from common.config.opts import opts


def main(opt):
    # 设置随机种子
    torch.manual_seed(opt.seed)

    # 设置训练设备
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

    # 加载训练集
    DataTrain = COCO(opt, 'train')
    base_s = DataTrain.coco
    train_loader = torch.utils.data.DataLoader(
        DataTrain,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True
    )
    # 加载验证集
    val_intervals = opt.val_intervals
    DataVal = COCO(opt, 'val')
    val_loader = torch.utils.data.DataLoader(
        DataVal,
        batch_size=1,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True
    )

    # 加载模型
    print('Creating model...')
    heads = {'hm': DataTrain.num_classes, 'wh': 2, 'reg': 2}
    model = DSFNet(heads)
    # 设置优化器
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    # 初始化
    start_epoch = 0
    if not os.path.exists(opt.save_dir):
        os.mkdir(opt.save_dir)
    if not os.path.exists(opt.save_results_dir):
        os.mkdir(opt.save_results_dir)
    logger = Logger(opt)
    # 加载训练过的模型权重
    if opt.load_model != '':
        model, optimizer, start_epoch = load_model_weight(model, opt.load_model, optimizer, opt.resume,
                                                          opt.lr, opt.lr_step)
    # 训练器
    trainer = CtdetTrainer(opt, model, optimizer)
    trainer.set_device(opt.gpus, opt.device)

    print('Starting training...')

    best = -1
    for epoch in range(start_epoch + 1, opt.num_epochs + 1):

        log_dict_train, _ = trainer.train(epoch, train_loader)

        logger.write('epoch: {} |'.format(epoch))

        save_model(os.path.join(opt.save_dir, 'model_last.pth'),
                   epoch, model, optimizer)

        for k, v in log_dict_train.items():
            logger.write('{} {:8f} | '.format(k, v))
        if val_intervals > 0 and epoch % val_intervals == 0:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                       epoch, model, optimizer)
            with torch.no_grad():
                log_dict_val, preds, stats = trainer.val(epoch, val_loader,base_s, DataVal)
            for k, v in log_dict_val.items():
                logger.write('{} {:8f} | '.format(k, v))
            logger.write('eval results: ')
            for k in stats.tolist():
                logger.write('{:8f} | '.format(k))
            if log_dict_val['ap50'] > best:
                best = log_dict_val['ap50']
                save_model(os.path.join(opt.save_dir, 'model_best.pth'),
                           epoch, model)
        else:
            save_model(os.path.join(opt.save_dir, 'model_last.pth'),
                       epoch, model, optimizer)
        logger.write('\n')
        if epoch in opt.lr_step:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
                       epoch, model, optimizer)
            lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
            print('Drop LR to', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    logger.close()

if __name__ == '__main__':
    opt = opts().parse()
    main(opt)