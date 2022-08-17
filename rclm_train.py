from utils.utils_data import *
from utils.utils_algo import *
# from data_utils import *
from models import *
import argparse, time, os
from torchvision import transforms
import torch
from torch import nn

parser = argparse.ArgumentParser(
    prog='complementary-label learning demo file.',
    usage='Demo with complementary labels.',
    description='A simple demo file with MNIST dataset.',
    epilog='end',
    add_help=True)

parser.add_argument('-lr', '--learning_rate', help='optimizer\'s learning rate', default=5e-6, type=float)
parser.add_argument('-bs', '--batch_size', help='batch_size of ordinary labels.', default=40, type=int)
parser.add_argument('-me', '--method', help='method type. non_k_softmax: only equation (5) . w_loss: weighted loss.',
                    choices=['w_loss', 'non_k_softmax'], default='w_loss', type=str)  # required=True)
parser.add_argument('-mo', '--model', help='model name', choices=['vgg16', 'res152', 'incv3', 'incresv2'], default='vgg16', type=str)
parser.add_argument('-e', '--epochs', help='number of epochs', type=int, default=5)
parser.add_argument('-wd', '--weight_decay', help='weight decay', default=1e-4, type=float)
parser.add_argument('-is', '--image_size', help='image size', default=224, type=int)

args = parser.parse_args()

# fix random seed as 0
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# prepare imagenet dataset
full_train_loader, train_loader, test_loader, ordinary_train_dataset, test_dataset, K = prepare_imagenet_data(
    batch_size=args.batch_size, image_size=args.image_size)

# model architecture
model = Vgg_16()  # Vgg_16, Res_152, Inc_v3, IncRes_v2
# initialization
for m in model.modules():
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_normal_(m.weight)

model = model.to(device)
model.eval()
internal = [i for i in range(29)]

# instantiate optimizer
optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.learning_rate)
save_table = np.zeros(shape=(args.epochs, 3))

for epoch in range(args.epochs):
    train_loss = 0
    print("epoch:" + str(epoch))
    # train of each epoch
    for i, (images, labels) in enumerate(train_loader):
        if i % 100 == 0:
            print(i)
        resize = transforms.Resize([args.image_size, args.image_size])
        images = resize(images)
        images, labels = images.to(device), labels.to(device)
        _, outputs = model.prediction(images)
        loss = chosen_loss_c(f=outputs, K=1000, labels=labels, method=args.method)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss = train_loss + loss.item()

    train_accuracy = accuracy_check(loader=train_loader, model=model)
    test_accuracy = accuracy_check(loader=test_loader, model=model)
    print('Epoch: {}. Tr Acc: {}. Te Acc: {}.'.format(epoch + 1, train_accuracy, test_accuracy))
    save_table[epoch, :] = epoch + 1, train_accuracy, test_accuracy

torch.save(model.state_dict(), 'rclm/' + args.model + '_rclm.pt')
np.savetxt('rclm/' + args.model + '_rclm.txt', save_table, delimiter=',', fmt='%1.3f')
