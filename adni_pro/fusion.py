import torch
import torch.nn as nn
import torch.nn.functional as F
from graph_learning.module.modules.layers.classifier import MLP
from graph_learning.module import ModuleConfig, get_module
from graph_learning.utils import merge_dicts, dict_merge_rec

def prepare_hiddens(xs):
    xs_ret, masks_ret = [], []
    for x in xs:
        if isinstance(x, list):
            x, mask = prepare_hiddens(x)
        else:
            mask = None
            if isinstance(x, tuple):
                if len(x) == 2:
                    x, mask = x
                elif len(x) == 3:
                    x, _, mask = x
            if mask is None:
                mask = torch.ones(x.size(0), dtype=torch.bool, device=x.device)
            if len(mask.shape)==1:
                mask = mask.unsqueeze(1)

        if isinstance(x, list):
            xs_ret = xs_ret + x
        else:
            xs_ret.append(x)

        masks_ret.append(mask)
    mask = torch.cat(masks_ret, -1)
    return xs_ret, mask

class FusionModuleConfig(ModuleConfig):
    def __init__(self, args, context):
        super().__init__(args, context)
        self.encoders = [get_module(context, n)
                         for n in self.encoders]

    @property
    def builder(self):
        return FusionModule

    @classmethod
    def define_parser(cls, parser):
        super().define_parser(parser)
        parser.add_argument('--encoders', nargs='+')
        parser.add_argument('--freeze', action='store_true')

class FusionModule(nn.Module):
    def __init__(self, encoders, freeze=False):
        super().__init__()
        self.encoders = nn.ModuleList(encoders)
        if freeze:
            for p in self.encoders.parameters():
                p.requires_grad = False

    def fusion(self, *xs):
        raise NotImplementedError

    def forward(self, data, feature):
        encoder_outputs = [encoder(data, feature)
                           for encoder in self.encoders]
        xs = [eo['hidden'] for eo in encoder_outputs]
        outputs = merge_dicts([eo['outputs'] for eo in encoder_outputs])
        fusion_outputs = self.fusion(*xs)
        hidden = fusion_outputs['hidden']
        outputs = dict_merge_rec(outputs, fusion_outputs['outputs'])
        return {'hidden': hidden,
                'outputs': outputs}

@ModuleConfig.register('mfb')
class MFBFusionConfig(FusionModuleConfig):
    @property
    def builder(self):
        return MFB

    @classmethod
    def define_parser(cls, parser):
        super().define_parser(parser)
        parser.add_argument('--in-sizes', type=int, nargs='+')
        parser.add_argument('--hidden-size', type=int)
        parser.add_argument('--dropout', type=float)
        parser.add_argument('--aux-hiddens', action='store_true')
        parser.add_argument('--k', type=int)

class MFB(FusionModule):
    def __init__(self, encoders,
                 in_sizes, hidden_size, k, dropout,
                 aux_hiddens):
        super().__init__(encoders)
        assert len(in_sizes) == 2

        self.aux_hiddens = aux_hiddens
        self.k = k

        self.proj_1 = nn.Linear(in_sizes[0], hidden_size*k)
        self.proj_2 = nn.Linear(in_sizes[1], hidden_size*k)
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.AvgPool1d(k, stride=k)

    def fusion(self, *xs):
        xs, mask = prepare_hiddens(xs)

        h1, h2 = xs

        batch_size = h1.size(0)
        h1 = self.proj_1(h1)[:, None, :]
        h2 = self.proj_2(h2)[:, None, :]

        exp_out = h1 * h2
        exp_out = self.dropout(exp_out)
        z = self.pool(exp_out) * self.k
        z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
        z = F.normalize(z[:, 0])

        mask_f = torch.ones(mask.size(0), 1, dtype=torch.bool, device=mask.device)
        mask = torch.cat([mask_f, mask], -1)

        hidden_ret = ([z, *xs], None, mask) if self.aux_hiddens else z
        hs_tsne = {'h_fusion': z}

        return {'hidden': hidden_ret,
                'outputs': {'hs_tsne': hs_tsne}}

@ModuleConfig.register('plain-fusion')
class PlainFusionConfig(FusionModuleConfig):
    @property
    def builder(self):
        return PlainFusion

    @classmethod
    def define_parser(cls, parser):
        super().define_parser(parser)
        parser.add_argument('--typ')
        parser.add_argument('--aux-hiddens', action='store_true')

class PlainFusion(FusionModule):
    def __init__(self, encoders, typ, aux_hiddens, freeze):
        super().__init__(encoders, freeze)
        self.typ = typ
        self.aux_hiddens = aux_hiddens

    def fusion(self, *xs):
        xs, mask = prepare_hiddens(xs)

        if self.typ == 'cat':
            h = torch.cat(xs, -1)
        mask_f = torch.ones(mask.size(0), 1, dtype=torch.bool, device=mask.device)
        hidden_ret = ([h, *xs], None, torch.cat([mask_f, mask], -1)) if self.aux_hiddens else h

        hs_tsne = {'h_fusion': h}
        return {'hidden': hidden_ret,
                'outputs': {'hs_tsne': hs_tsne}}

@ModuleConfig.register('raw-fusion')
class RawFusionConfig(FusionModuleConfig):
    @property
    def builder(self):
        return RawFusion

    @classmethod
    def define_parser(cls, parser):
        super().define_parser(parser)

class RawFusion(FusionModule):
    def __init__(self, encoders, freeze):
        super().__init__(encoders, freeze)

    def fusion(self, *xs):
        xs, masks = prepare_hiddens(xs)
        mask = torch.stack(masks, -1)

        hidden_ret = (xs, None, mask)

        return {'hidden': hidden_ret,
                'outputs': {}}

@ModuleConfig.register('gmu')
class GMUModuleConfig(FusionModuleConfig):
    def __init__(self, args, context):
        super().__init__(args, context)

    @property
    def builder(self):
        return GMU

    @classmethod
    def define_parser(cls, parser):
        super().define_parser(parser)
        parser.add_argument('--in-sizes', type=int, nargs='+')
        parser.add_argument('--hidden-size', type=int)
        parser.add_argument('--dropout', type=float)
        parser.add_argument('--aux-hiddens', action='store_true')

class GMU(FusionModule):
    def __init__(self, encoders,
                 in_sizes, hidden_size, dropout,
                 aux_hiddens):
        super().__init__(encoders)
        self.aux_hiddens = aux_hiddens

        k = len(in_sizes)

        self.fc_xs = nn.ModuleList()
        self.fc_gs = nn.ModuleList()
        for i in range(k):
            self.fc_xs.append(nn.Linear(in_sizes[i], hidden_size))
            self.fc_gs.append(nn.Linear(in_sizes[i], k))

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def fusion(self, *xs):
        xs, mask = prepare_hiddens(xs)
        assert len(xs) == len(self.fc_xs)

        hs = [self.dropout(self.relu(fc_x(x)))
              for x, fc_x in zip(xs, self.fc_xs)]
        hs = torch.stack(hs, 1)

        gs = [fc_g(x)
              for x, fc_g in zip(xs, self.fc_gs)]
        gate = sum(gs)
        gate[~mask] = float('-inf')
        gate = F.gumbel_softmax(gate, dim=-1)

        h = torch.einsum('bkh,bk->bh', hs, gate)

        #nodeviz = {f'w_{i}':gate[:, i] for i in range(gate.size(1))}
        nodeviz = {}
        hs_tsne = {'h_fusion': h}
        hs_viz = {}
        #hs_viz = {'h_fusion': h}

        hidden_ret = ([h, *xs], gate, mask) if self.aux_hiddens else h

        return {'hidden': hidden_ret,
                'outputs': {'hs_tsne': hs_tsne,
                            'hiddens': hs_viz,
                            'nodeviz': nodeviz}}

class SymFusion(nn.Module):
    def __init__(self, in_size, hidden_size, dropout):
        super().__init__()
        self.f1 = nn.Linear(in_size, hidden_size)
        self.f2 = nn.Linear(in_size, hidden_size)

        self.g1 = nn.Linear(in_size, 1)
        self.g2 = nn.Linear(in_size, 1)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
    def forward(self, h1, h2):
        h1_o = self.dropout(self.tanh(self.f1(h1)))
        h2_o = self.dropout(self.tanh(self.f2(h2)))
        g1 = self.sigmoid(self.g1(h1))
        g2 = self.sigmoid(self.g2(h2))
        hidden = h1_o * g2 + h2_o * g1
        #hidden = torch.cat([h1_o * g2, h2_o * g1], -1)
        return hidden, torch.cat([g2, g1], -1)

class MLB(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, dropout):
        super().__init__()
        self.f1 = nn.Linear(in_size, hidden_size)
        self.f2 = nn.Linear(in_size, hidden_size)
        self.fo = nn.Linear(hidden_size, out_size)
        self.activation = nn.Tanh()
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x1, x2):
        h1 = self.activate(self.f1(x1))
        h2 = self.activate(self.f2(x2))
        out = h1 * h2
        out = self.dropout_layer(out)
        return self.fo(out)

@ModuleConfig.register('netgated')
class NetGatedModuleConfig(FusionModuleConfig):
    def __init__(self, args, context):
        super().__init__(args, context)

    @property
    def builder(self):
        return NetGated

    @classmethod
    def define_parser(cls, parser):
        super().define_parser(parser)
        parser.add_argument('--in-sizes', type=int, nargs='+')
        parser.add_argument('--hidden-size', type=int)
        parser.add_argument('--dropout', type=float)
        parser.add_argument('--aux-hiddens', action='store_true')

class NetGated(FusionModule):
    def __init__(self, encoders,
                 in_sizes, hidden_size, dropout,
                 aux_hiddens):
        super().__init__(encoders)
        self.aux_hiddens = aux_hiddens

        k = len(in_sizes)

        self.fc_xs = nn.ModuleList()
        for i in range(k):
            self.fc_xs.append(nn.Linear(in_sizes[i], hidden_size))

        self.fc_g = MLP(k*hidden_size, hidden_size, k, 1, dropout, False)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def fusion(self, *xs):
        assert len(xs) == len(self.fc_xs)
        xs, masks = prepare_hiddens(xs)
        hs = [self.dropout(self.relu(fc_x(x)))
              for x, fc_x in zip(xs, self.fc_xs)]
        hs = torch.stack(hs, 1)

        gate = self.fc_g(hs.reshape(hs.size(0), -1))

        mask = torch.stack(masks, -1)
        #gate[~mask] = float('-inf')
        gate = F.softmax(gate, -1)

        h = torch.einsum('bkh,bk->bh', hs, gate)
        nodeviz = {f'w_{i}':gate[:, i] for i in range(gate.size(1))}
        hs_tsne = {'h_fusion': h}
        hs_viz = {'h_fusion': h}

        hidden_ret = ([h, *xs], gate, mask) if self.aux_hiddens else h

        return {'hidden': hidden_ret,
                'outputs': {'hs_tsne': hs_tsne,
                            'hiddens': hs_viz,
                            'nodeviz': nodeviz}}

@ModuleConfig.register('mag')
class MAGModuleConfig(FusionModuleConfig):
    def __init__(self, args, context):
        super().__init__(args, context)

    @property
    def builder(self):
        return MAG

    @classmethod
    def define_parser(cls, parser):
        super().define_parser(parser)
        parser.add_argument('--in-sizes', type=int, nargs='+')
        parser.add_argument('--dropout', type=float)
        parser.add_argument('--aux-hiddens', action='store_true')
        parser.add_argument('--beta-shift', type=float, default=1)

class MAG(FusionModule):
    def __init__(self, encoders,
                 in_sizes, dropout, beta_shift,
                 aux_hiddens):
        super().__init__(encoders)
        self.aux_hiddens = aux_hiddens
        self.beta_shift = beta_shift

        self.fc_gs = nn.ModuleList()
        self.fc_hs = nn.ModuleList()
        in_size_main = in_sizes[0]
        for in_size in in_sizes[1:]:
            self.fc_gs.append(nn.Linear(in_size_main+in_size, in_size_main))
            self.fc_hs.append(nn.Linear(in_size, in_size_main))

        self.layer_norm = nn.LayerNorm(in_sizes[0])
        self.dropout = nn.Dropout(dropout)

    def fusion(self, *xs):
        assert len(xs) == len(self.fc_hs) + 1
        xs, masks = prepare_hiddens(xs)
        mask = torch.stack(masks, -1)

        eps = 1e-6
        x_main = xs[0]
        gates = [F.relu(fc_g(torch.cat([x_main, x], -1)))
                 for x, fc_g in zip(xs[1:], self.fc_gs)]
        hs = [fc_h(x)
              for x, fc_h in zip(xs[1:], self.fc_hs)]

        hm = sum([h*g for h, g in zip(hs, gates)])

        em_norm = x_main.norm(2, dim=-1)
        hm_norm = hm.norm(2, dim=-1)

        hm_norm_ones = torch.ones_like(hm_norm, requires_grad=True)
        hm_norm = torch.where(hm_norm==0, hm_norm_ones, hm_norm)

        thresh_hold = (em_norm/(hm_norm+eps)) * self.beta_shift
        ones = torch.ones_like(thresh_hold, requires_grad=True)

        alpha = torch.min(thresh_hold, ones).unsqueeze(-1)

        h_aux = alpha * hm
        h = self.dropout(
            #x_main + h_aux
            self.layer_norm(x_main + h_aux)
        )

        hs_tsne = {'h_fusion': h}
        #hs_viz = {'h_fusion': h, 'fusion_gate': torch.cat(gates, -1)}
        hs_viz = {'fusion_gate': torch.cat(gates, -1)}

        hidden_ret = ([h, *xs[1:]], None, mask) if self.aux_hiddens else h

        return {'hidden': hidden_ret,
                'outputs': {'hs_tsne': hs_tsne,
                            'hiddens': hs_viz,
                            'nodeviz': {}}}

@ModuleConfig.register('moe-weighting')
class MoEWModuleConfig(FusionModuleConfig):
    def __init__(self, args, context):
        super().__init__(args, context)

    @property
    def builder(self):
        return MoEW

    @classmethod
    def define_parser(cls, parser):
        super().define_parser(parser)
        parser.add_argument('--in-sizes', type=int, nargs='+')
        parser.add_argument('--hidden-size', type=int)

class MoEW(FusionModule):
    def __init__(self, encoders,
                 in_sizes, hidden_size, freeze):
        super().__init__(encoders)
        if freeze:
            for p in self.encoders.parameters():
                p.requires_grad = False

        k = len(in_sizes)

        self.fc_gs = nn.ModuleList()
        for i in range(k):
            self.fc_gs.append(nn.Linear(in_sizes[i], k))

    def fusion(self, *xs):
        xs, mask = prepare_hiddens(xs)
        assert len(xs) == len(self.fc_gs)
        gs = [fc_g(x) for x, fc_g in zip(xs, self.fc_gs)]
        gate = sum(gs)
        gate[~mask] = float('-inf')
        #gate = F.softmax(gate, dim=-1)
        gate = F.gumbel_softmax(gate, dim=-1)
        #gate = torch.where(torch.isnan(gate), torch.zeros_like(gate), gate)

        hidden_ret = (xs, gate, mask)

        return {'hidden': hidden_ret,
                'outputs': {}}

@ModuleConfig.register('moddrop')
class ModDropModuleConfig(FusionModuleConfig):
    def __init__(self, args, context):
        super().__init__(args, context)

    @property
    def builder(self):
        return ModDrop

    @classmethod
    def define_parser(cls, parser):
        super().define_parser(parser)
        parser.add_argument('--drop-rate', type=float)

class ModDrop(FusionModule):
    def __init__(self, encoders, drop_rate):
        super().__init__(encoders)
        self.drop_rate = drop_rate

    def fusion(self, *xs):
        xs, mask = prepare_hiddens(xs)
        if self.training:
            binomial = torch.distributions.binomial.Binomial(probs=1-self.drop_rate)
            keep_masks = [binomial.sample((x.size(0), 1)).to(x.device)
                          for x in xs]
            xs = [x * keep_mask
                  for x, keep_mask in zip(xs, keep_masks)]

            keep_mask = torch.cat(keep_masks, -1)
            mask = (mask * keep_mask).bool()
            #mask = torch.ones_like(mask)

        return {'hidden': (xs, None, mask),
                'outputs': {}}
