import torch


def print_model_parm_flops(model, input, detail=False):
    list_conv = []

    def conv_hook(self, input, output):

        # batch_size, input_channels, input_time(ops) ,input_height, input_width = input[0].size()
        # output_channels,output_time(ops) , output_height, output_width = output[0].size()

        kernel_ops = (self.in_channels / self.groups) * 2 - 1  # add operations is one less to the mul operations
        for i in self.kernel_size:
            kernel_ops *= i
        bias_ops = 1 if self.bias is not None else 0

        params = kernel_ops + bias_ops
        flops = params * output[0].nelement()

        list_conv.append(flops)

    list_linear = []

    def linear_hook(self, input, output):
        weight_ops = (2 * self.in_features - 1) * output.nelement()
        bias_ops = self.bias.nelement()
        flops = weight_ops + bias_ops
        list_linear.append(flops)

    list_bn = []

    def bn_hook(self, input, output):
        # (x-x')/σ one sub op and one div op
        # and the shift γ and β
        list_bn.append(input[0].nelement() / input[0].size(0) * 4)

    list_relu = []

    def relu_hook(self, input, output):
        # every input's element need to cmp with 0
        list_relu.append(input[0].nelement() / input[0].size(0))

    list_pooling = []

    def max_pooling_hook(self, input, output):
        # batch_size, input_channels, input_height, input_width = input[0].size()
        # output_channels, output_height, output_width = output[0].size()

        # unlike conv ops. in pool layer ,if the kernel size is a int ,self.input will be a int,not a tuple.
        # so we need to deal with this problem
        if isinstance(self.kernel_size, tuple):
            kernel_ops = torch.prod(torch.Tensor([self.kernel_size]))
        else:
            kernel_ops = self.kernel_size * self.kernel_size
            if len(output[0].size()) > 3:  # 3D max pooling
                kernel_ops *= self.kernel_size
        flops = kernel_ops * output[0].nelement()
        list_pooling.append(flops)

    def avg_pooling_hook(self, input, output):
        # cmp to max pooling ,avg pooling has an additional sub op
        # unlike conv ops. in pool layer ,if the kernel size is a int ,self.input will be a int,not a tuple.
        # so we need to deal with this problem
        if isinstance(self.kernel_size, tuple):
            kernel_ops = torch.prod(torch.Tensor([self.kernel_size]))
        else:
            kernel_ops = self.kernel_size * self.kernel_size
            if len(output[0].size()) > 3:  # 3D  pooling
                kernel_ops *= self.kernel_size
        flops = (kernel_ops + 1) * output[0].nelement()
        list_pooling.append(flops)

    def adaavg_pooling_hook(self, input, output):
        kernel = torch.Tensor([*(input[0].shape[2:])]) // torch.Tensor(list((self.output_size,))).squeeze()
        kernel_ops = torch.prod(kernel)
        flops = (kernel_ops + 1) * output[0].nelement()
        list_pooling.append(flops)

    def adamax_pooling_hook(self, input, output):
        kernel = torch.Tensor([*(input[0].shape[2:])]) // torch.Tensor(list((self.output_size,))).squeeze()
        kernel_ops = torch.prod(kernel)
        flops = kernel_ops * output[0].nelement()
        list_pooling.append(flops)

    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d) or isinstance(net, torch.nn.Conv3d):
                net.register_forward_hook(conv_hook)
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
            if isinstance(net, torch.nn.BatchNorm2d) or isinstance(net, torch.nn.BatchNorm3d):
                net.register_forward_hook(bn_hook)
            if isinstance(net, torch.nn.ReLU):
                net.register_forward_hook(relu_hook)
            if isinstance(net, torch.nn.MaxPool2d) or isinstance(net, torch.nn.MaxPool3d):
                net.register_forward_hook(max_pooling_hook)
            if isinstance(net, torch.nn.AvgPool2d) or isinstance(net, torch.nn.AvgPool3d):
                net.register_forward_hook(avg_pooling_hook)
            if isinstance(net, torch.nn.AdaptiveAvgPool2d) or isinstance(net, torch.nn.AdaptiveAvgPool3d):
                net.register_forward_hook(adaavg_pooling_hook)
            if isinstance(net, torch.nn.AdaptiveMaxPool2d) or isinstance(net, torch.nn.AdaptiveMaxPool3d):
                net.register_forward_hook(adamax_pooling_hook)
            return
        for c in childrens:
            foo(c)

    foo(model)
    out = model(input)
    total_flops = sum(list_conv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling)
    print(' + Number of FLOPs: %.2fG' % (total_flops / 1e9))

    if detail:
        print('  + Conv FLOPs: %.2fG' % (sum(list_conv) / 1e9))
        print('  + Linear FLOPs: %.2fG' % (sum(list_linear) / 1e9))
        print('  + Batch Norm FLOPs: %.2fG' % (sum(list_bn) / 1e9))
        print('  + Relu FLOPs: %.2fG' % (sum(list_relu) / 1e9))
        print('  + Pooling FLOPs: %.2fG' % (sum(list_pooling) / 1e9))


if __name__ == '__main__':
    class net(torch.nn.Module):
        def __init__(self):
            super(net, self).__init__()
            self.conv1 = torch.nn.Conv3d(3, 5, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool = torch.nn.AvgPool3d((2, 2, 1))

        def forward(self, x):
            x = self.conv1(x)
            x = self.pool(x)
            return x


    from torchvision import models
    model = models.resnet50()
    input = torch.randn(1, 3, 224, 224)
    print_model_parm_flops(model, input, detail=True)

    from slowfastnet import SlowFastNet

    model = SlowFastNet()
    input = torch.randn(1, 3, 48, 224, 224)
    print_model_parm_flops(model, input, detail=True)
