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
            self.alpha=Variable(torch.tensor([0.00,50,1,1,20]))
        else:
            if isinstance(alpha,Variable):
                self.alpha=alpha
            else:
                self.alpha=Variable(alpha)
        # self.alpha=self.alpha.cuda()

    def forward(self,outputs,targets):
        # N for batchsize
        # C for class size
        N = outputs.size(0)
        C = outputs.size(1)
        P = outputs.softmax(dim=1)

        # 将target转成one-hot编码
        class_mask = outputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        # outputs.is_cuda
        # self.alpha.is_cuda

        if outputs.is_cuda and not self.alpha.is_cuda:
            self.alpha=self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]
        probs = (P * class_mask).sum(1).view(-1, 1)
        log_p=probs.log()
        batch_loss=-alpha*torch.pow((1-probs),self.gamma)*log_p

        if self.size_average:
            loss=batch_loss.mean()
        else:
            loss=batch_loss.sum()
        return loss





if __name__ == "__main__":
    output = torch.tensor([[0.9, 0, 0, 0, 0], [0, 0.89, 0, 0, 0]])
    target = torch.tensor([0, 2])
    lossfun=FocalLoss()
    loss=lossfun(output,target)
    print("focal loss:",loss)