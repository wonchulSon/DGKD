import os
import copy
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from data_loader import get_cifar
from model_factory import create_cnn_model, is_resnet
import random

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    else:
        return False


def parse_arguments():
    parser = argparse.ArgumentParser(description='TA Knowledge Distillation Code')
    parser.add_argument('--epochs', default=160, type=int, help='number of total epochs to run')
    parser.add_argument('--dataset', default='cifar100', type=str, help='dataset. can be either cifar10 or cifar100')
    parser.add_argument('--crop', default=False, type=str2bool, help='augmentation Ture or False')
    parser.add_argument('--batch-size', default=128, type=int, help='batch_size')
    parser.add_argument('--learning-rate', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='SGD momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float, help='SGD weight decay (default: 1e-4)')

    parser.add_argument('--T', default=5, type=int, help='T')
    parser.add_argument('--seed', default=20, type=int, help='seed')
    parser.add_argument('--lamb', default=1, type=float, help='lambda')

    parser.add_argument('--teacher', default='plane10', type=str, help='teacher name')
    parser.add_argument('--ta1', default='plane8', type=str)
    parser.add_argument('--ta2', default='plane6', type=str)
    parser.add_argument('--ta3', default='plane4', type=str)

    parser.add_argument('--teacher-checkpoint', default='/path', type=str)
    parser.add_argument('--ta1-checkpoint', default='/path', type=str)
    parser.add_argument('--ta2-checkpoint', default='/path', type=str)
    parser.add_argument('--ta3-checkpoint', default='/path', type=str)

    parser.add_argument('--student', default='plane2', type=str, help='student name')
    parser.add_argument('--TA-count', default=3, type=int, help='TA count')

    parser.add_argument('--cuda', default=True, type=str2bool, help='whether or not use cuda(train on GPU)')
    parser.add_argument('--gpus', default='0', type=str, help='Which GPUs you want to use? (0,1,2,3)')
    parser.add_argument('--drop-num', default=1, type=int, help='random drop')
    parser.add_argument('--dataset-dir', default='./data', type=str, help='dataset directory')
    args = parser.parse_args()
    return args


def load_checkpoint(model, checkpoint_path):
    model_ckp = torch.load(checkpoint_path)
    model.load_state_dict(model_ckp['model_state_dict'])
    return model


class TrainManager(object):
    def __init__(self, student, teacher=None, ta_list=None, train_loader=None, test_loader=None, train_config={}):
        self.student = student
        self.teacher = teacher
        for i, ta in enumerate(ta_list):
            globals()["self.ta{}".format(i + 1)] = ta

        self.have_teacher = bool(self.teacher)
        self.device = train_config['device']
        self.name = train_config['name']
        self.optimizer = optim.SGD(self.student.parameters(),
                                   lr=train_config['learning_rate'],
                                   momentum=train_config['momentum'],
                                   weight_decay=train_config['weight_decay'])
        self.teacher.eval()
        self.teacher.train(mode=False)
        for i, ta in enumerate(ta_list):
            globals()["self.ta{}".format(i + 1)].eval()
            globals()["self.ta{}".format(i + 1)].train(mode=False)

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = train_config

    def train(self):
        lambda_ = self.config['lambda_student']
        T = self.config['T_student']
        epochs = self.config['epochs']
        drop_num = self.config['drop_num']

        iteration = 0
        best_acc = 0
        criterion = nn.CrossEntropyLoss()
        for epoch in range(epochs):
            self.student.train()
            self.adjust_learning_rate(self.optimizer, epoch)
            loss = 0
            for batch_idx, (data, target) in enumerate(self.train_loader):
                iteration += 1
                data = data.to(self.device)
                target = target.to(self.device)
                self.optimizer.zero_grad()
                student_output = self.student(data)

                # Standard Learning Loss (Classification Loss)
                loss_SL = criterion(student_output, target)

                teacher_outputs = self.teacher(data)
                ta_outputs = []
                for i in range(len(ta_list)):
                    ta_outputs.append(globals()["self.ta{}".format(i + 1)](data))

                # Teacher Knowledge Distillation Loss
                loss_KD_list = [nn.KLDivLoss()(F.log_softmax(student_output / T, dim=1),
                                               F.softmax(teacher_outputs / T, dim=1))]

                # Teacher Assistants Knowledge Distillation Loss
                for i in range(len(ta_list)):
                    loss_KD_list.append(nn.KLDivLoss()(F.log_softmax(student_output / T, dim=1),
                                                       F.softmax(ta_outputs[i] / T, dim=1)))

                # Stochastic DGKD
                if args.drop_num != 0:
                    for _ in range(args.drop_num):
                        loss_KD_list.remove(random.choice(loss_KD_list))

                # Total Loss
                loss = (1 - lambda_) * loss_SL + lambda_ * T * T * sum(loss_KD_list)

                loss.backward()
                self.optimizer.step()

            print("epoch {}/{}".format(epoch, epochs))
            val_acc = self.validate(step=epoch)
            if val_acc > best_acc:
                best_acc = val_acc
                print('**** best val acc: ' + str(best_acc) + ' ****')
                self.save(epoch, name='DGKD_{}_{}_best.pth.tar'.format(args.gpus, self.name, args.dataset))
            print('loss: ', loss.data)
            print()

        return best_acc

    def validate(self, step=0):
        self.student.eval()
        with torch.no_grad():
            total = 0
            correct = 0

            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                output = self.student(images)

                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            acc = 100 * correct / total

            return acc

    def save(self, epoch, name=None):
        torch.save({
            'model_state_dict': self.student.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,
        }, name)


    def adjust_learning_rate(self, optimizer, epoch):
        epochs = self.config['epochs']
        models_are_plane = self.config['is_plane']

        # depending on dataset
        if models_are_plane:
            lr = 0.01
        else:
            if epoch < int(epochs / 2.0):
                lr = 0.1
            elif epoch < int(epochs * 3 / 4.0):
                lr = 0.1 * 0.1
            else:
                lr = 0.1 * 0.01

        # update optimizer's learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


if __name__ == "__main__":
    # Parsing arguments and prepare settings for training
    args = parse_arguments()
    print(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    torch.cuda.manual_seed(args.seed)

    dataset = args.dataset
    num_classes = 100 if dataset == 'cifar100' else 10

    print("---------- Creating Students -------")
    student_model = create_cnn_model(args.student, dataset, use_cuda=args.cuda)

    train_config = {
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        'device': 'cuda' if args.cuda else 'cpu',
        'is_plane': not is_resnet(args.student),
        'T_student': args.T,
        'lambda_student': args.lamb,
        'drop_num': args.drop_num,
    }

    # Train Teacher if provided a teacher, otherwise it's a normal training using only cross entropy loss
    # This is for training single models for baselines models (or training the first teacher)
    if args.teacher:
        teacher_model = create_cnn_model(args.teacher, dataset, use_cuda=args.cuda)
        if args.teacher_checkpoint:
            print("---------- Loading Teacher -------")
            teacher_model = load_checkpoint(teacher_model, args.teacher_checkpoint)
        else:
            print("---------- Training Teacher -------")
            train_loader, test_loader = get_cifar(num_classes)
            teacher_train_config = copy.deepcopy(train_config)
            teacher_name = '{}_best.pth.tar'.format(args.teacher)
            teacher_train_config['name'] = args.teacher
            teacher_trainer = TrainManager(teacher_model, teacher=None, train_loader=train_loader,
                                           test_loader=test_loader, train_config=teacher_train_config)
            teacher_trainer.train()
            teacher_model = load_checkpoint(teacher_model, os.path.join('./', teacher_name))

    # Prepare Teacher and Assistants
    print("---------- Creating Model ----------")
    teacher_model = create_cnn_model(args.teacher, dataset, use_cuda=args.cuda)
    models_dict = {}
    for i in range(1, args.TA_num + 1):
        models_dict['model{}'.format(i)] = create_cnn_model(getattr(args, 'ta{}'.format(i)), dataset, use_cuda=args.cuda)

    print("---------- Loading Model ----------")
    teacher_model = load_checkpoint(teacher_model, args.teacher_checkpoint)
    ta_list=[]
    for i in range(1, args.TA_num + 1):
        ta_list.append(load_checkpoint(models_dict['model{}'.format(i)], getattr(args, 'ta{}_checkpoint'.format(i))))

    # Student training
    print("---------- Training Student -------")
    student_train_config = copy.deepcopy(train_config)
    train_loader, test_loader = get_cifar(num_classes,  crop=args.crop)
    student_train_config['name'] = args.student
    student_trainer = TrainManager(student_model, teacher=teacher_model, ta_list=ta_list,
                                   train_loader=train_loader,
                                   test_loader=test_loader,
                                   train_config=student_train_config)

    best_student_acc = student_trainer.train()
