#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 09:07:10 2019

@author: xingyu
"""
import os
from model.LPRNET import LPRNet, CHARS
from model.STN import STNet
from data.load_data import LPRDataLoader, collate_fn
from Evaluation import eval, decode
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import copy
import numpy as np
import argparse
import time


def sparse_tuple_for_ctc(T_length, lengths):
    input_lengths = []
    target_lengths = []

    for ch in lengths:
        input_lengths.append(T_length)
        target_lengths.append(ch)

    return tuple(input_lengths), tuple(target_lengths)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='LPR Training')
    parser.add_argument('--img_size', default=(94, 24), help='the image size')
    parser.add_argument('--img_dirs_train', default="/data/zjw/data/ccpd/train/", help='the training images path')
    parser.add_argument('--img_dirs_val', default="/data/zjw/data/ccpd/val/", help='the validation images path')
    parser.add_argument('--dropout_rate', default=0.5, help='dropout rate.')
    parser.add_argument('--epoch', type=int, default=100, help='number of epoches for training')
    parser.add_argument('--batch_size', default=32, help='batch size')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    lprnet = LPRNet(class_num=len(CHARS), dropout_rate=args.dropout_rate)
    lprnet.to(device)
    lprnet.load_state_dict(torch.load('/data/zjw/code/LPD/LPRNet/weights/green_lprnet_model.pth', map_location=lambda storage, loc: storage))
    print("LPRNet loaded")
    
    STN = STNet()
    STN.to(device)
    STN.load_state_dict(torch.load('/data/zjw/code/LPD/LPRNet/weights/green_STN_model.pth', map_location=lambda storage, loc: storage))
    print("STN loaded")
    
    dataset = {'train': LPRDataLoader([args.img_dirs_train], args.img_size),
               'val': LPRDataLoader([args.img_dirs_val], args.img_size)}
    dataloader = {'train': DataLoader(dataset['train'], batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn),
                  'val': DataLoader(dataset['val'], batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)}
    print('training dataset loaded with length : {}'.format(len(dataset['train'])))
    print('validation dataset loaded with length : {}'.format(len(dataset['val'])))
    
    # define optimizer & loss
    optimizer = torch.optim.Adam([{'params': STN.parameters(), 'weight_decay': 2e-5},
                                  {'params': lprnet.parameters()}])
    ctc_loss = nn.CTCLoss(blank=len(CHARS)-1, reduction='mean') # reduction: 'none' | 'mean' | 'sum'
    ## save logging and weights
    train_logging_file = '/data/zjw/code/LPD/LPRNet/weights/train_logging.txt'
    validation_logging_file = '/data/zjw/code/LPD/LPRNet/weights/validation_logging.txt'
    save_dir = '/data/zjw/code/LPD/LPRNet/weights/saving_ckpt'
    # if os.path.exists(save_dir):
    #     raise NameError('model dir exists!')
    # os.makedirs(save_dir)
    
    start_time = time.time()
    total_iters = 0
    best_acc = 0.0
    T_length = 18 # args.lpr_max_len
    print('training kicked off..')
    print('-' * 10) 
    for epoch in range(args.epoch):
        # train model
        lprnet.train()
        STN.train()
        since = time.time()
        for imgs, labels, lengths in dataloader['train']:   # img: torch.Size([2, 3, 24, 94])  # labels: torch.Size([14]) # lengths: [7, 7] (list)
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            with torch.set_grad_enabled(True):
                transfer = STN(imgs)
                logits = lprnet(transfer)  # torch.Size([batch_size, CHARS length, output length ])
                log_probs = logits.permute(2, 0, 1) # for ctc loss: length of output x batch x length of chars
                log_probs = log_probs.log_softmax(2).requires_grad_()       
                input_lengths, target_lengths = sparse_tuple_for_ctc(T_length, lengths) # convert to tuple with length as batch_size 
                loss = ctc_loss(log_probs, labels, input_lengths=input_lengths, target_lengths=target_lengths)
                
                loss.backward()
                optimizer.step()
                
                total_iters += 1
                # print train information
                if total_iters % 100 == 0:
                    # current training accuracy             
                    preds = logits.cpu().detach().numpy()  # (batch size, 68, 18)
                    _, pred_labels = decode(preds, CHARS)  # list of predict output
                    total = preds.shape[0]
                    start = 0
                    TP = 0
                    for i, length in enumerate(lengths):
                        label = labels[start:start+length]
                        start += length
                        if np.array_equal(np.array(pred_labels[i]), label.cpu().numpy()):
                            TP += 1
                    
                    time_cur = (time.time() - since) / 100
                    since = time.time()
                    
                    for p in  optimizer.param_groups:
                        lr = p['lr']
                    print("Epoch {}/{}, Iters: {:0>6d}, loss: {:.4f}, train_accuracy: {:.4f}, time: {:.2f} s/iter, learning rate: {}"
                          .format(epoch, args.epoch-1, total_iters, loss.item(), TP/total, time_cur, lr))
                    with open(train_logging_file, 'a') as f:
                        f.write("Epoch {}/{}, Iters: {:0>6d}, loss: {:.4f}, train_accuracy: {:.4f}, time: {:.2f} s/iter, learning rate: {}"
                          .format(epoch, args.epoch-1, total_iters, loss.item(), TP/total, time_cur, lr)+'\n')
                    f.close()

                    lprnet.eval()
                    STN.eval()
                    ACC = eval(lprnet, STN, dataloader['val'], dataset['val'], device)
                    if best_acc <= ACC:
                        best_acc = ACC
                        best_model_lprnet = copy.deepcopy(lprnet.state_dict())
                        best_model_stn = copy.deepcopy(STN.state_dict())
                    print("Epoch {}/{}, Iters: {:0>6d}, validation_accuracy: {:.4f}".format(epoch, args.epoch - 1,
                                                                                            total_iters, ACC))
                    with open(validation_logging_file, 'a') as f:
                        f.write("Epoch {}/{}, Iters: {:0>6d}, validation_accuracy: {:.4f}".format(epoch, args.epoch - 1,
                                                                                                  total_iters,
                                                                                                  ACC) + '\n')
                    f.close()
                    lprnet.train()
                    STN.train()

                    # save model
            # if total_iters % 900 == 0:
            #
            #     torch.save({
            #         'iters': total_iters,
            #         'net_state_dict': lprnet.state_dict()},
            #         os.path.join(save_dir, 'lprnet_Iter_%06d_model.ckpt' % total_iters))
            #
            #     torch.save({
            #         'iters': total_iters,
            #         'net_state_dict': STN.state_dict()},
            #         os.path.join(save_dir, 'stn_Iter_%06d_model.ckpt' % total_iters))
                    
            # evaluate accuracy
            # best_iters = total_iters
            # if total_iters % 900 == 0:
            #
            #     lprnet.eval()
            #     STN.eval()
            #
            #     ACC = eval(lprnet, STN, dataloader['val'], dataset['val'], device)
            #
            #     if best_acc <= ACC:
            #         best_acc = ACC
            #         best_iters = total_iters
            #
            #     print("Epoch {}/{}, Iters: {:0>6d}, validation_accuracy: {:.4f}".format(epoch, args.epoch-1, total_iters, ACC))
            #     with open(validation_logging_file, 'a') as f:
            #         f.write("Epoch {}/{}, Iters: {:0>6d}, validation_accuracy: {:.4f}".format(epoch, args.epoch-1, total_iters, ACC)+'\n')
            #     f.close()
            #
            #     lprnet.train()
            #     STN.train()
                                
    time_elapsed = time.time() - start_time
    STN.load_state_dict(best_model_stn)
    torch.save(STN.state_dict(), '/data/zjw/code/LPD/LPRNet/weights/final_green_STN_model.pth')
    lprnet.load_state_dict(best_model_lprnet)
    torch.save(lprnet.state_dict(), '/data/zjw/code/LPD/LPRNet/weights/final_green_lprnet_model.pth')
    print('Finally Best Accuracy: {:.4f}'.format(best_acc))
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))