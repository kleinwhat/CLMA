""" Script for CLMA. """
import sys
import numpy as np
import torch

sys.path.append('/home/yantao/workspace/projects/baidu/CVPR2019_workshop')

device = torch.device('cuda:0')
torch.set_default_tensor_type(torch.FloatTensor)

def vgg_normalization(image):
    return image - [123.68, 116.78, 103.94]

def inception_normalization(image):
    return ((image / 255.) - 0.5) * 2

class ComplementaryLabelAttack(object):
    """ CLMA, using pytorch."""

    def __init__(self, src_model, rcl_model, epsilon=16., step_size=1.6, steps=10, decay_factor=1, ens=30, prob=0.7):
        self.step_size = step_size
        self.epsilon = epsilon
        self.steps = steps
        self.src_model = src_model
        self.decay_factor = decay_factor
        self.rcl_model = rcl_model
        self.ens = ens
        self.prob = prob

    def __call__(self, X_nat, y, index, attack_layer_idx=-1, internal=[]):
        X_nat_np = X_nat.cpu().numpy()
        X_nat_np = np.transpose(X_nat_np,(0,2,3,1))
        # vgg_normalization for for Vgg-16 and Res-152, inception_normalization for inc-v3, inc-res-v2
        X_nat_np = vgg_normalization(np.copy(X_nat_np))
        X_nat_np = np.transpose(X_nat_np,(0,3,1,2))

        for p in self.src_model.parameters():
            p.requires_grad = False
        for p in self.rcl_model.parameters():
            p.requires_grad = False
        
        self.rcl_model.eval()
        self.src_model.eval()
        X_adv = np.copy(X_nat_np)
        X = np.copy(X_nat_np)
        X_ten_adv = torch.from_numpy(X_adv).float()
        X_ten_adv = X_ten_adv.to(device)
        weight = 0
        for l in range(int(self.ens)):
            np.random.seed(0)
            mask = np.random.binomial(1, self.prob, size=(X_nat.shape[0],X_nat.shape[1],X_nat.shape[2],X_nat.shape[3]))
            images_tmp_np = X_nat_np * mask
            images_tmp_np = np.transpose(images_tmp_np,(0,2,3,1))
            # vgg_normalization for for Vgg-16 and Res-152, inception_normalization for inc-v3, inc-res-v2
            images_tmp_np = vgg_normalization(np.copy(images_tmp_np))
            images_tmp_np = np.transpose(images_tmp_np,(0,3,1,2))
            images_tmp_np2 = torch.from_numpy(images_tmp_np).float()
            images_tmp_np2 = images_tmp_np2.to(device)
            images_tmp_np2.requires_grad = True
            images_tmp_np2.retain_grad()
            rclm_features, rclm_pred = self.rcl_model.prediction(images_tmp_np2, internal=internal)
            rclm_feature = rclm_features[attack_layer_idx]
            rclm_feature.retain_grad()
            output = rclm_pred * y
            output.backward(torch.ones_like(output))
            weight += rclm_feature.grad * (-1)
        torch.save(weight, 'FIA/weight/weight_tensor' + str(index) + '.pt')
        momentum = 0
        for i in range(self.steps):
            X_ten_adv = X_ten_adv.to(device)
            X_ten_adv.requires_grad = True
            X_ten_adv.retain_grad()
            internal_features, pred = self.src_model.prediction(X_ten_adv, internal=internal)
            feature = internal_features[attack_layer_idx]
            loss = ((torch.mul(feature, weight)).sum())
            self.src_model.zero_grad()
            loss.backward(torch.ones_like(loss))

            grad = X_ten_adv.grad.data.cpu().numpy()
            velocity = grad / np.mean(np.absolute(grad))
            X_ten_adv.grad.zero_()
            momentum = self.decay_factor * momentum + velocity

            X -= self.step_size * np.sign(momentum)
            X = np.clip(X, X_nat_np - self.epsilon, X_nat_np + self.epsilon)
            X_ten_adv = torch.FloatTensor(X)
        return torch.from_numpy(X)
