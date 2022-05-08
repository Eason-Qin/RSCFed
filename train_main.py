from validation import epochVal_metrics_test
from options import args_parser
import os
import sys
import logging
import random
import numpy as np
import copy
import datetime
from FedAvg import FedAvg, model_dist
import torch
from torchvision import transforms
import torch.backends.cudnn as cudnn
from networks.models import ModelFedCon
from dataloaders import dataset
from local_supervised import SupervisedLocalUpdate
from local_unsupervised import UnsupervisedLocalUpdate
from tqdm import trange
from cifar_load import get_dataloader, partition_data, partition_data_allnoniid
from torch.utils.tensorboard import SummaryWriter


def split(dataset, num_users):
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def test(epoch, checkpoint, data_test, label_test, n_classes):

    net = ModelFedCon(args.model, args.out_dim, n_classes=n_classes)
    if len(args.gpu.split(',')) > 1:
        net = torch.nn.DataParallel(net, device_ids=[i for i in range(round(len(args.gpu) / 2))])
    model = net.cuda()
    model.load_state_dict(checkpoint)

    if args.dataset == 'SVHN' or args.dataset == 'cifar100':
        test_dl, test_ds = get_dataloader(args, data_test, label_test,
                                          args.dataset, args.datadir, args.batch_size,
                                          is_labeled=True, is_testing=True)
    elif args.dataset == 'skin':
        test_dl, test_ds = get_dataloader(args, data_test, label_test,
                                          args.dataset, args.datadir, args.batch_size,
                                          is_labeled=True, is_testing=True, pre_sz=args.pre_sz, input_sz=args.input_sz)

    AUROCs, Accus = epochVal_metrics_test(model, test_dl, args.model, thresh=0.4, n_classes=n_classes)
    AUROC_avg = np.array(AUROCs).mean()
    Accus_avg = np.array(Accus).mean()

    return AUROC_avg, Accus_avg

if __name__ == '__main__':
    args = args_parser()

    # meta info for clients
    # 1 supervised 
    supervised_user_id = [0]
    # 9 unsupervised
    unsupervised_user_id = list(range(len(supervised_user_id), args.unsup_num + len(supervised_user_id)))

    sup_num = len(supervised_user_id)
    unsup_num = len(unsupervised_user_id)
    total_num = sup_num + unsup_num

    # for loggers
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    time_current = 'attempt0'
    if args.log_file_name is None:
        args.log_file_name = 'log-%s' % (datetime.datetime.now().strftime("%m-%d-%H%M-%S"))
    log_path = args.log_file_name + '.log'
    logging.basicConfig(filename=os.path.join(args.logdir, log_path), level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.info(str(args))
    logger.info(time_current)
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    if not os.path.isdir('tensorboard'):
        os.mkdir('tensorboard')

    # dataset configs
    if args.dataset == 'SVHN':
        if not os.path.isdir('tensorboard/SVHN/' + time_current):
            os.mkdir('tensorboard/cares_SVHN/' + time_current)
        writer = SummaryWriter('tensorboard/SVHN/' + time_current)

    elif args.dataset == 'cifar100':
        if not os.path.isdir('tensorboard/cifar100/' + time_current):
            os.mkdir('tensorboard/cifar100/' + time_current)
        writer = SummaryWriter('tensorboard/cifar100/' + time_current)

    elif args.dataset == 'skin':
        if not os.path.isdir('tensorboard/skin/' + time_current):
            os.mkdir('tensorboard/skin/' + time_current)
        writer = SummaryWriter('tensorboard/skin/' + time_current)

    snapshot_path = 'model/'
    if not os.path.isdir(snapshot_path):
        os.mkdir(snapshot_path)
    if args.dataset == 'SVHN':
        snapshot_path = 'model/SVHN/'
    if args.dataset == 'cifar100':
        snapshot_path = 'model/cifar100/'
    if args.dataset == 'skin':
        snapshot_path = 'model/skin/'
    if not os.path.isdir(snapshot_path):
        os.mkdir(snapshot_path)


    print('==> Reloading data partitioning strategy..')
    assert os.path.isdir('partition_strategy'), 'Error: no partition_strategy directory found!'

    if args.dataset == 'SVHN':
        partition = torch.load('partition_strategy/SVHN_noniid_10%labeled.pth')
        net_dataidx_map = partition['data_partition']
    elif args.dataset == 'cifar100':
        partition = torch.load('partition_strategy/cifar100_noniid_10%labeled.pth')
        net_dataidx_map = partition['data_partition']

    # partitioning dataset in non-iid fashion using dirichlet distribution
    X_train, y_train, X_test, y_test, _, traindata_cls_counts = partition_data_allnoniid(
        args.dataset, args.datadir, partition=args.partition, n_parties=total_num, beta=args.beta)

    if args.dataset == 'SVHN':
        X_train = X_train.transpose([0, 2, 3, 1])
        X_test = X_test.transpose([0, 2, 3, 1])

    if args.dataset == 'cifar10' or args.dataset == 'SVHN':
        n_classes = 10
    elif args.dataset == 'cifar100':
        n_classes = 100
    elif args.dataset == 'skin':
        n_classes = 7

    # definition of global model
    net_glob = ModelFedCon(args.model, args.out_dim, n_classes=n_classes)

    if args.resume:
        print('==> Resuming from checkpoint..')
        if args.dataset == 'cifar100':
            checkpoint = torch.load('warmup/cifar100.pth')
        elif args.dataset == 'SVHN':
            checkpoint = torch.load('warmup/SVHN.pth')

        net_glob.load_state_dict(checkpoint['state_dict'])
        start_epoch = 7
    else:
        start_epoch = 0

    if len(args.gpu.split(',')) > 1:
        net_glob = torch.nn.DataParallel(net_glob, device_ids=[i for i in range(round(len(args.gpu) / 2))])  #
    net_glob.train()

    # for distribution to local clients
    w_glob = net_glob.state_dict()
    # storages for local clients' weights
    w_locals = []
    # unsupervised teacher models' weights
    w_ema_unsup = []
    # trainer for un/labelled clients
    lab_trainer_locals = []
    unlab_trainer_locals = []
    # net for sup/unsup
    sup_net_locals = []
    unsup_net_locals = []
    # optimizers for sup/unsup
    sup_optim_locals = []
    unsup_optim_locals = []

    dist_scale_f = args.dist_scale

    # total data amount
    total_lenth = sum([len(net_dataidx_map[i]) for i in range(len(net_dataidx_map))])

    # each client's data amount
    each_lenth = [len(net_dataidx_map[i]) for i in range(len(net_dataidx_map))]

    # foreach each_lenth's data proportion 
    client_freq = [len(net_dataidx_map[i]) / total_lenth for i in range(len(net_dataidx_map))]

    # deploy trainers for sup clients
    for i in supervised_user_id:
        lab_trainer_locals.append(SupervisedLocalUpdate(args, net_dataidx_map[i], n_classes))
        w_locals.append(copy.deepcopy(w_glob))
        sup_net_locals.append(copy.deepcopy(net_glob))
        if args.opt == 'adam':
            optimizer = torch.optim.Adam(sup_net_locals[i].parameters(), lr=args.base_lr,
                                         betas=(0.9, 0.999), weight_decay=5e-4)
        elif args.opt == 'sgd':
            optimizer = torch.optim.SGD(sup_net_locals[i].parameters(), lr=args.base_lr, momentum=0.9,
                                        weight_decay=5e-4)
        elif args.opt == 'adamw':
            optimizer = torch.optim.AdamW(sup_net_locals[i].parameters(), lr=args.base_lr, weight_decay=0.02)
        if args.resume:
            optimizer.load_state_dict(checkpoint['sup_optimizers'][i])
        sup_optim_locals.append(copy.deepcopy(optimizer.state_dict()))

    # deploy trainers for unsup clients
    for i in unsupervised_user_id:
        unlab_trainer_locals.append(
            UnsupervisedLocalUpdate(args, net_dataidx_map[i], n_classes))
        # locals--students
        w_locals.append(copy.deepcopy(w_glob))
        # with ema--teachers
        w_ema_unsup.append(copy.deepcopy(w_glob))
        unsup_net_locals.append(copy.deepcopy(net_glob))
        if args.opt == 'adam':
            optimizer = torch.optim.Adam(unsup_net_locals[i - sup_num].parameters(), lr=args.unsup_lr,
                                         betas=(0.9, 0.999), weight_decay=5e-4)
        elif args.opt == 'sgd':
            optimizer = torch.optim.SGD(unsup_net_locals[i - sup_num].parameters(),
                                        lr=args.unsup_lr, momentum=0.9,
                                        weight_decay=5e-4)
        elif args.opt == 'adamw':
            optimizer = torch.optim.AdamW(unsup_net_locals[i - sup_num].parameters(), lr=args.unsup_lr,
                                          weight_decay=0.02)
        if args.resume and len(checkpoint['unsup_optimizers']) != 0:
            optimizer.load_state_dict(checkpoint['unsup_optimizers'][i - sup_num])
        unsup_optim_locals.append(copy.deepcopy(optimizer.state_dict()))

        if args.resume and len(checkpoint['unsup_ema_state_dict']) != 0 and not args.from_labeled:
            w_ema_unsup = copy.deepcopy(checkpoint['unsup_ema_state_dict'])
            unlab_trainer_locals[i - sup_num].ema_model.load_state_dict(w_ema_unsup[i - sup_num])
            unlab_trainer_locals[i - sup_num].flag = False
            print('Unsup EMA reloaded')

    # FL officially starting from there
    for com_round in trange(start_epoch, args.rounds):
        print("************* Comm round %d begins *************" % com_round)

        # clients' loss
        loss_locals = []
        # clients sampled for the round
        clt_this_comm_round = []
        # sub-consensus models' weights
        w_per_meta = []

        # each meta_round for a sub-consensus model

        # "The number of subsampling operations M and the number of local clients 
        # used in each sub-sampling operation K are set as: M =3, and K=5." //from Paper Sec.4
        for meta_round in range(args.meta_round):
            # sample 5 clients
            clt_list_this_meta_round = random.sample(list(range(0, total_num)), args.meta_client_num)
            # record selected clients
            clt_this_comm_round.extend(clt_list_this_meta_round)
            # record the sup client if selected.
            chosen_sup = [j for j in supervised_user_id if j in clt_list_this_meta_round]
            logger.info(f'Comm round {com_round} meta round {meta_round} chosen client: {clt_list_this_meta_round}')
            # record weights for this subcon- 
            w_locals_this_meta_round = []

            # local training
            for client_idx in clt_list_this_meta_round:
                # supervised one old-fashioned
                if client_idx in supervised_user_id:
                    local = lab_trainer_locals[client_idx]
                    optimizer = sup_optim_locals[client_idx]
                    train_dl_local, train_ds_local = get_dataloader(args, X_train[net_dataidx_map[client_idx]],
                                                                    y_train[net_dataidx_map[client_idx]],
                                                                    args.dataset, args.datadir, args.batch_size,
                                                                    is_labeled=True,
                                                                    data_idxs=net_dataidx_map[client_idx],
                                                                    pre_sz=args.pre_sz, input_sz=args.input_sz)
                    w, loss, op = local.train(args, sup_net_locals[client_idx].state_dict(), optimizer,
                                                           train_dl_local, n_classes)  # network, loss, optimizer
                    writer.add_scalar('Supervised loss on sup client %d' % client_idx, loss, global_step=com_round)
                    w_locals_this_meta_round.append(copy.deepcopy(w))
                    sup_optim_locals[client_idx] = copy.deepcopy(op)
                    loss_locals.append(copy.deepcopy(loss))
                    logger.info(
                        'Labeled client {} sample num: {} training loss : {} lr : {}'.format(client_idx,
                                                                                             len(train_ds_local),
                                                                                             loss,
                                                                                             sup_optim_locals[
                                                                                                 client_idx][
                                                                                                 'param_groups'][0][
                                                                                                 'lr']))
                
                # unsupervised -- mean teacher | with ema-> teacher | without ema-> student | students' weights for next step
                else:
                    local = unlab_trainer_locals[client_idx - sup_num]
                    optimizer = unsup_optim_locals[client_idx - sup_num]
                    train_dl_local, train_ds_local = get_dataloader(args,
                                                                    X_train[net_dataidx_map[client_idx]],
                                                                    y_train[net_dataidx_map[client_idx]],
                                                                    args.dataset,
                                                                    args.datadir, args.batch_size, is_labeled=False,
                                                                    data_idxs=net_dataidx_map[client_idx],
                                                                    pre_sz=args.pre_sz, input_sz=args.input_sz)
                    # trained for student weights ready to use
                    w, w_ema, loss, op, ratio, correct_pseu, all_pseu, test_right, train_right, test_right_ema, same_pred_num = local.train(
                        args,
                        unsup_net_locals[client_idx - sup_num].state_dict(),
                        optimizer,
                        com_round * args.local_ep,
                        client_idx,
                        train_dl_local, n_classes)
                    writer.add_scalar('Unsupervised loss on unsup client %d' % client_idx, loss, global_step=com_round)

                    # record student weights
                    w_locals_this_meta_round.append(copy.deepcopy(w))
                    # record corresp. teacher weights
                    w_ema_unsup[client_idx - sup_num] = copy.deepcopy(w_ema)
                    # record optimizers
                    unsup_optim_locals[client_idx - sup_num] = copy.deepcopy(op)
                    # record client loss
                    loss_locals.append(copy.deepcopy(loss))
                    logger.info(
                        'Unlabeled client {} sample num: {} Training loss: {}, unsupervised loss ratio: {}, lr {}, {} pseu out of {} are correct, {} correct by model, {} correct by ema before train, {} by model during train, total {}'.format(
                            client_idx, len(train_ds_local), loss,
                            ratio,
                            unsup_optim_locals[
                                client_idx - sup_num][
                                'param_groups'][
                                0]['lr'], correct_pseu, all_pseu, test_right, test_right_ema, train_right,
                            len(net_dataidx_map[client_idx])))

            # data amount for each client in this subset
            each_lenth_this_meta_round = [each_lenth[clt] for clt in clt_list_this_meta_round]
            # replicate
            each_lenth_this_meta_raw = copy.deepcopy(each_lenth_this_meta_round)
            # total_lenth_this_meta = sum(each_lenth_this_meta_round)

            # following *Federated semi-supervised learning for covid region segmentation in chest ct using multi-national data from china, italy, japan*
            # "Hence, we re-implement [28], try increased aggregation weight for labeled client from the set {20%, 30%, 50%, 70%}." //from Paper Sec. 4.2
            if args.w_mul_times != 1 and 0 in clt_list_this_meta_round and (
                    args.un_dist == '' or args.un_dist_onlyunsup):  # and com_round<=40:  # :
                for sup_idx in chosen_sup:
                    each_lenth_this_meta_round[clt_list_this_meta_round.index(sup_idx)] *= args.w_mul_times

            # total data amount in this subset 
            # N_total in Eqn.5
            total_lenth_this_meta = sum(each_lenth_this_meta_round)
            # fractions in Eqn.5
            clt_freq_this_meta_round = [i / total_lenth_this_meta for i in each_lenth_this_meta_round]
            print('Based on data amount: ' + f'{clt_freq_this_meta_round}')
            clt_freq_this_meta_raw = copy.deepcopy(clt_freq_this_meta_round)

            # implementation of Eqn.5 | Alg.1 line5
            w_avg_temp = FedAvg(w_locals_this_meta_round, clt_freq_this_meta_round)

            # DMA implementation of Eqn.6 | Alg.1 line6
            dist_list = []
            for cli_idx in range(args.meta_client_num):
                dist = model_dist(w_locals_this_meta_round[cli_idx], w_avg_temp) # L2 norm distance in the num of exp 
                dist_list.append(dist)
            print(
                'Normed dist * 1e4 : ' + f'{[dist_list[i] * 1e5 / each_lenth_this_meta_raw[i] for i in range(args.meta_client_num)]}')
            

            # left part of Eqn.6 + weightedmul on sup cli
            if len(chosen_sup) != 0:
                clt_freq_this_meta_uncer = [
                    np.exp(-dist_list[i] * args.sup_scale / each_lenth_this_meta_raw[i]) * clt_freq_this_meta_round[i] for i
                    in
                    range(args.meta_client_num)]
                for sup_idx in chosen_sup:
                    mul_times = args.w_mul_times
                    clt_freq_this_meta_uncer[clt_list_this_meta_round.index(
                            sup_idx)] *= mul_times  # (args.w_mul_times/len(chosen_sup))
            else:
                clt_freq_this_meta_uncer = [
                    np.exp(-dist_list[i] * dist_scale_f / each_lenth_this_meta_raw[i]) * clt_freq_this_meta_round[i]
                    for i
                    in range(args.meta_client_num)]

            # right part of Eqn.6 | normalize the the intra-subset model weight
            total = sum(clt_freq_this_meta_uncer)
            clt_freq_this_meta_dist = [clt_freq_this_meta_uncer[i] / total for i in range(args.meta_client_num)]
            clt_freq_this_meta_round = clt_freq_this_meta_dist
            print('After dist-based uncertainty : ' + f'{clt_freq_this_meta_round}')

            assert sum(clt_freq_this_meta_round) - 1.0 <= 1e-3, "Error: sum(freq) != 0"

            # Alg.1 line7
            w_this_meta = FedAvg(w_locals_this_meta_round, clt_freq_this_meta_round)
            
            # append this subconsensus model
            w_per_meta.append(w_this_meta)

        # Alg.1 line8 | Eqn.7  fedavg all subconsesus model
        each_lenth_this_round = [each_lenth[clt] for clt in clt_this_comm_round]

        if args.w_mul_times != 1 and 0 in clt_this_comm_round:
            each_lenth_this_round[clt_this_comm_round.index(0)] *= args.w_mul_times
        total_lenth_this = sum(each_lenth_this_round)
        clt_freq_this_round = [i / total_lenth_this for i in each_lenth_this_round]

        with torch.no_grad():
            freq = [1 / args.meta_round for i in range(args.meta_round)]
            w_glob = FedAvg(w_per_meta, freq)

        # update and load and test
        net_glob.load_state_dict(w_glob)
        for i in supervised_user_id:
            sup_net_locals[i].load_state_dict(w_glob)
        for i in unsupervised_user_id:
            unsup_net_locals[i - sup_num].load_state_dict(w_glob)

        loss_avg = sum(loss_locals) / len(loss_locals)
        logger.info(
            '************ Loss Avg {}, LR {}, Round {} ends ************  '.format(loss_avg, args.base_lr, com_round))
        if com_round % 6 == 0:
            if not os.path.isdir(snapshot_path + time_current):
                os.mkdir(snapshot_path + time_current)
            save_mode_path = os.path.join(snapshot_path + time_current, 'epoch_' + str(com_round) + '.pth')
            if len(args.gpu) != 1:
                torch.save({
                    'state_dict': net_glob.module.state_dict(),
                    'unsup_ema_state_dict': w_ema_unsup,
                    'sup_optimizers': sup_optim_locals,
                    'unsup_optimizers': unsup_optim_locals,
                    'start_epoch': com_round
                }
                    , save_mode_path
                )
            else:
                torch.save({
                    'state_dict': net_glob.state_dict(),
                    'unsup_ema_state_dict': w_ema_unsup,
                    'sup_optimizers': sup_optim_locals,
                    'unsup_optimizers': unsup_optim_locals,
                    'start_epoch': com_round
                }
                    , save_mode_path
                )
        AUROC_avg, Accus_avg = test(com_round, net_glob.state_dict(), X_test, y_test, n_classes)
        writer.add_scalar('AUC', AUROC_avg, global_step=com_round)
        writer.add_scalar('Acc', Accus_avg, global_step=com_round)
        logger.info("\nTEST Student: Epoch: {}".format(com_round))
        logger.info("\nTEST AUROC: {:6f}, TEST Accus: {:6f}"
                    .format(AUROC_avg, Accus_avg))
