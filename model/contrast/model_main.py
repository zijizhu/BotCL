from timm.models import create_model
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from model.contrast.slots import ScouterAttention, vis
from model.contrast.position_encode import build_position_encoding

def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)

class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding2D, self).__init__()
        print('Using custom PE2D')
        self.org_channels = channels
        channels = int(np.ceil(channels / 4) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cached_penc", None, persistent=False)

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, y, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device, dtype=self.inv_freq.dtype)
        pos_y = torch.arange(y, device=tensor.device, dtype=self.inv_freq.dtype)
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = get_emb(sin_inp_x).unsqueeze(1)
        emb_y = get_emb(sin_inp_y)
        emb = torch.zeros(
            (x, y, self.channels * 2),
            device=tensor.device,
            dtype=tensor.dtype,
        )
        emb[:, :, : self.channels] = emb_x
        emb[:, :, self.channels : 2 * self.channels] = emb_y

        self.cached_penc = emb[None, :, :, :orig_ch].repeat(tensor.shape[0], 1, 1, 1)
        return self.cached_penc


class Identical(nn.Module):
    def __init__(self):
        super(Identical, self).__init__()

    def forward(self, x):
        return x


def load_backbone(args):
    bone = create_model(args.base_model, pretrained=True,
                        num_classes=args.num_classes)

    if args.dataset == "MNIST":
            bone.conv1 = nn.Conv2d(1, 64, 3, stride=2, padding=1, bias=False)
    bone.global_pool = Identical()
    bone.fc = Identical()
    # fix_parameter(bone, [""], mode="fix")
    # fix_parameter(bone, ["layer4", "layer3"], mode="open")
    return bone


class MainModel(nn.Module):
    def __init__(self, args, vis=False):
        super(MainModel, self).__init__()
        self.args = args
        self.pre_train = args.pre_train
        if "18" not in args.base_model:
            self.num_features = 2048
        else:
            self.num_features = 512
        self.feature_size = args.feature_size
        self.drop_rate = 0.0
        hidden_dim = 128
        num_concepts = args.num_cpt
        num_classes = args.num_classes
        self.back_bone = create_model(args.base_model, pretrained=True, num_classes=args.num_classes)
        self.activation = nn.Tanh()
        self.vis = vis

        if not self.pre_train:
            self.conv1x1 = nn.Conv2d(self.num_features, hidden_dim, kernel_size=(1, 1), stride=(1, 1))
            self.norm = nn.BatchNorm2d(hidden_dim)
            self.position_emb = PositionalEncoding2D(hidden_dim)
            self.slots = ScouterAttention(args, hidden_dim, num_concepts, vis=self.vis)
            self.scale = 1
            self.cls = torch.nn.Linear(num_concepts, num_classes)
        else:
            self.fc = nn.Linear(self.num_features, num_classes)
            self.drop_rate = 0

    def forward(self, x, weight=None, things=None):
        x = self.back_bone.forward_features(x)
        features = x
        # x = x.view(x.size(0), self.num_features, self.feature_size, self.feature_size)

        if not self.pre_train:
            x = self.conv1x1(x)
            x = self.norm(x)
            x = torch.relu(x)
            pe = self.position_emb(x)
            x_pe = x + pe

            b, n, r, c = x.shape
            x = x.reshape((b, n, -1)).permute((0, 2, 1))
            x_pe = x_pe.reshape((b, n, -1)).permute((0, 2, 1))
            updates, attn = self.slots(x_pe, x, weight, things)
            if self.args.cpt_activation == "att":
                cpt_activation = attn
            else:
                cpt_activation = updates
            attn_cls = self.scale * torch.sum(cpt_activation, dim=-1)
            cpt = self.activation(attn_cls)
            cls = self.cls(cpt)
            return (cpt - 0.5) * 2, cls, attn, updates
        else:
            x = F.adaptive_max_pool2d(x, 1).squeeze(-1).squeeze(-1)
            if self.drop_rate > 0:
                x = F.dropout(x, p=self.drop_rate, training=self.training)
            x = self.fc(x)
            return x, features


class MainModel2(nn.Module):
    def __init__(self, args, vis=False):
        super(MainModel2, self).__init__()
        self.args = args
        self.pre_train = args.pre_train
        if "18" not in args.base_model:
            self.num_features = 2048
        else:
            self.num_features = 512
        self.feature_size = args.feature_size
        self.drop_rate = 0.0
        self.back_bone = load_backbone(args)

    def forward(self, x):
        x = self.back_bone(x)
        features = x

        return features


# if __name__ == '__main__':
#     model = MainModel()
#     inp = torch.rand((2, 1, 224, 224))
#     pred, out, att_loss = model(inp)
#     print(pred.shape)
#     print(out.shape)