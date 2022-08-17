import torch
import torch.nn.functional as F
from torchvision import transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
internal = [i for i in range(29)]


def non_k_softmax_loss(f, K, labels):
    Q_1 = 1 - F.softmax(f, 1)
    Q_1 = F.softmax(Q_1, 1)
    labels = labels.long()
    return F.nll_loss(Q_1.log(), labels.long())  # Equation(8) in paper


def w_loss(f, K, labels):
    loss_class = non_k_softmax_loss(f=f, K=K, labels=labels)
    loss_w = w_loss_p(f=f, K=K, labels=labels)
    final_loss = loss_class + loss_w  # Equation(11) in paper
    return final_loss


# weighted loss
def w_loss_p(f, K, labels):
    Q_1 = 1 - F.softmax(f, 1)
    Q = F.softmax(Q_1, 1)
    q = torch.tensor(1.0) / torch.sum(Q_1, dim=1)
    q = q.view(-1, 1).repeat(1, K)
    w = torch.mul(Q_1, q)  # weight
    w_1 = torch.mul(w, Q.log())
    return F.nll_loss(w_1, labels.long())  # Equation(14) in paper


# choose loss
def chosen_loss_c(f, K, labels, method):
    if method == 'non_k_softmax':
        final_loss = non_k_softmax_loss(f=f, K=K, labels=labels)
    elif method == 'w_loss':
        final_loss = w_loss(f=f, K=K, labels=labels)
    return final_loss


# check accuracy
def accuracy_check(loader, model):
    sm = F.softmax
    total, num_samples, cnt = 0, 0, 0
    for images, labels in loader:
        cnt = cnt + 1
        resize = transforms.Resize([299, 299])
        images = resize(images)
        labels = labels.to(device)
        images = images.to(device)
        _, outputs = model.prediction(images)
        sm_outputs = sm(outputs, dim=1)
        _, predicted = torch.max(sm_outputs.data, 1)
        total += (predicted == labels).sum().item()
        num_samples += labels.size(0)
    return 100 * total / num_samples
