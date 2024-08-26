import torch
from torch import nn

class BatchNorm(nn.Module):
    # num_features : number of outputs for a fully connected layer or a convolutional layer
    # num_dims: 2 for fully connected layer, 4 for convolutional layer 
    def __init__(self, num_features, num_dims, eps = 1e-5, momentum = 0.1):
        super().__init__()

        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)

        # scale and shift parameters (trained with backprop)
        self.gamma = torch.ones(shape) # initialized to ones
        self.beta = torch.zeros(shape) # initialized to zeros

        # trained with runnin momentum update
        self.running_mean = torch.zeros(shape) # initialized to zeros
        self.running_var = torch.ones(shape) # initialized to ones

        self.eps = eps
        self.momentum = momentum

    def forward(self, X):
        
        if not torch.is_grad_enabled():
            xmean = self.running_mean
            xvar = self.running_var
        else:
            if len(X.shape) == 2:
                xmean = X.mean(axis = 0)
                xvar = X.var(axis = 0)
            else:
                xmean = X.mean(axis = [0, 2, 3], keepdim = True)
                xvar = X.var(axis = [0, 2, 3], keepdim = True)   

            # update buffers
            with torch.no_grad():
                self.running_mean =  self.running_mean  * (1 - self.momentum) + self.momentum * xmean
                self.running_var = self.running_var * (1 - self.momentum) + self.momentum * xvar

        X_hat = (X - xmean) / torch.sqrt(xvar + self.eps)
        Y = self.gamma * X_hat + self.beta        
        return Y

class LayerNom(nn.Module):
    def __init__(self, num_features, eps = 1e-5):
        super().__init__()

        shape = (1, num_features)
        self.eps = eps

        # scale and shift parameters (trained with backprop)
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))

    def forward(self, X):
        mean = X.mean(axis = 1, keepdim = True)
        var = X.var(axis = 1, keepdim = True)

        X_hat = (X - mean) / (torch.sqrt(var) + self.eps)

        return self.gamma * X_hat + self.beta
        
class GroupNorm(nn.Module):
    def __init__(self, num_groups, num_channels, eps = 1e-5):
        super().__init__()

        assert num_channels % num_groups == 0, "Number of Channels should be evenly divisible by number of groups"
        
        self.eps = eps
        self.num_channels = num_channels
        self.num_groups = num_groups

        self.gamma = nn.Parameter(torch.ones(1, num_channels, 1, 1)) # (1, C, 1, 1)
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1)) # (1, C, 1, 1)

    def forward(self, X):

        X_shape = X.shape # (N, C, H, W)
        batch_size = X_shape[0]
        assert X.shape[1] == self.num_channels

        X = X.view(batch_size, self.num_groups, -1) # (N, G, (C/G)*H*W)

        mean = X.mean(axis = -1, keepdim = True) # (N, G, 1)
        var = X.var(axis = -1, keepdim = True) # (N, G, 1)

        X_hat = (X - mean) / torch.sqrt(var + self.eps) # (N, G, (C/G)*H*W)

        X_hat = X_hat.view(X_shape) # (N, C, H*W)

        Y = self.gamma * X_hat + self.beta
        
        return Y

class InstanceNorm(nn.Module):
    def __init__(self, num_channels, eps = 1e-5):
        super().__init__()

        self.eps = eps
        self.num_channels = num_channels

        self.gamma = nn.Parameter(torch.ones(1, num_channels, 1, 1)) # (1, C, 1, 1)
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1)) # (1, C, 1, 1)

    def forward(self, X):

        X_shape = X.shape # (N, C, H, W)
        batch_size = X_shape[0]
        assert X.shape[1] == self.num_channels

        X = X.view(batch_size, self.num_channels, -1) # (N, C, H*W)

        mean = X.mean(axis = -1, keepdim = True) # (N, C, 1)
        var = X.var(axis = -1, keepdim = True) # (N, C, 1)

        X_hat = (X - mean) / torch.sqrt(var + self.eps) 
        X_hat = X.view(X_shape)

        Y = self.gamma * X_hat + self.beta
        
        return Y

'''Batch Normalization'''
# convolutional
x = torch.randn(4, 3, 5, 5) 
bn = BatchNorm(3, 4)
y = bn(x)
#print(y[:, 0].mean(), y[:, 0].var())


# fully connected
x = torch.randn(4, 8) 
bn = BatchNorm(8, 2)
y = bn(x)
#print(y[:, 0].mean(), y[:, 0].var())


'''Layer Normalization'''
x = torch.randn(4, 8)
ln = LayerNom(8)
y = ln(x)
#print(y[0, :].mean(), y[0, :].var())


'''Group Normalization'''
x = torch.randn(4, 16, 10, 10)
gn = GroupNorm(4, 16)
y = gn(x)
#print(y[:, 0].mean(), y[:, 0].var())


'''Layer Normalization'''
x = torch.randn(4, 16, 10, 10)
inorm = InstanceNorm(16)
y = inorm(x)
#print(y[:, 0].mean(), y[:, 0].var())