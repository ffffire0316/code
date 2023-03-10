import torch
import torch.nn as nn
from torch.autograd import Variable

def CrossLoss(output,target):
    cr = nn.CrossEntropyLoss()
    return cr(output,target)

class FocalLoss(nn.Module):
    def __init__(self,alpha=None,gamma=2,size_average=True):
        super(FocalLoss,self).__init__()
        self.gamma=gamma
        self.size_average=size_average
        if alpha is None:
            self.alpha=Variable(torch.tensor([0,1,2,3,4])).float()
        else:
            if isinstance(alpha,Variable):
                self.alpha=alpha.float()
            else:
                self.alpha=Variable(alpha).float()
        # self.alpha=self.alpha.cuda()

    def forward(self,outputs,targets):
        # N for batchsize
        # C for class size
        N = outputs.size(0)
        C = outputs.size(1)
        P = outputs.softmax(dim=1).float()

        # 将target转成one-hot编码
        class_mask = outputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        # outputs.is_cuda
        # self.alpha.is_cuda

        if outputs.is_cuda and not self.alpha.is_cuda:
            self.alpha=self.alpha.cuda()
        # alpha_size (batchsize,) 1d
        alpha = self.alpha[ids.data.view(-1)]
        # probs size (batchsize,1) 2d
        probs = (P * class_mask).sum(1).view(-1, 1)
        log_p=probs.log()
        # nothing=alpha*log_p
        # something=torch.pow((1 - probs), self.gamma) * log_p
        # batch_loss=-alpha*torch.pow((1-probs),self.gamma)*log_p
        batch_loss=torch.dot(-alpha,(torch.pow((1 - probs), self.gamma) * log_p).squeeze())
        # dot_product = torch.dot(-alpha,something.squeeze())
        if self.size_average:
            # loss=batch_loss.mean()
            loss=batch_loss/N
        else:
            # loss=batch_loss.sum()
            loss=batch_loss
        return loss


def CapsLoss(y_pred,y_target):

    y_target = torch.zeros(y_target.size(0), 5).scatter_(1, y_target.view(-1, 1), 1.0)
    L = (
     y_target * torch.clamp(0.9 - y_pred, min=0.0) ** 2
     + 0.5 * (1 - y_target) * torch.clamp(y_pred - 0.1, min=0.0) ** 2
    )
    L_margin = L.sum(dim=1).mean()

    return L_margin




if __name__ == "__main__":
    output = torch.tensor([[0.9, 0, 0, 0, 0], [0, 0.89, 0, 0, 0],[0,0,3,1,1]])
    target = torch.tensor([0, 1, 2])
    lossfun=FocalLoss()
    loss=CapsLoss(output,target)
    print("focal loss:",loss)