import torch.nn.functional as F
from layers.tode import *
import torch
from layers.Embed import PatchEmbedding

class Wrapper(nn.Module):
    def __init__(self, module_list):
        super(Wrapper, self).__init__()
        self.layers = module_list

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, normalize_before=True, args_ode=None):
        super().__init__()

        self.normalize_before = normalize_before
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        assert args_ode.use_ode
        assert d_model == n_head * d_k
        assert d_model == n_head * d_v

        self.w_qs = InterpLinear(d_model, n_head * d_k, args_ode)
        self.w_ks = ODELinear(d_model, n_head * d_k, args_ode)
        self.w_vs = ODELinear(d_model, n_head * d_v, args_ode)

        self.fc = nn.Linear(d_v * n_head, d_model)
        nn.init.xavier_uniform_(self.fc.weight)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5, attn_dropout=dropout)

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, t, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q
        if self.normalize_before:
            q = self.layer_norm(q)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q, t).view(sz_b, len_q, len_q, -1, n_head, d_k)
        k = self.w_ks(k, t).view(sz_b, len_k, len_k, -1, n_head, d_k)
        v = self.w_vs(v, t).view(sz_b, len_v, len_v, -1, n_head, d_k)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.permute(0, 4, 1, 2, 3, 5), k.permute(0, 4, 1, 2, 3, 5), v.permute(0, 4, 1, 2, 3, 5)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        output, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))
        output += residual

        if not self.normalize_before:
            output = self.layer_norm(output)
        return output, attn

    def interpolate(self, q, k, v, t, qt, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        len_qt = qt.size(1)

        # t_clone = torch.cat((t, qt[:, -1:]), dim=-1)
        # q_clone = torch.cat((q, q[:, -1:, :]), dim=1)

        # coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(q_clone, t=t_clone[0])
        # spline = torchcde.CubicSpline(coeffs, t=t_clone[0])
        # residual = spline.evaluate(qt[0])

        residual = q[:, -1:, :].repeat(1, len_qt, 1)
        if self.normalize_before:
            q = self.layer_norm(q)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs.interpolate(q, t, qt).view(sz_b, len_qt, len_q, -1, n_head, d_k)
        k = self.w_ks.interpolate(k, t, qt).view(sz_b, len_qt, len_k, -1, n_head, d_k)
        v = self.w_vs.interpolate(v, t, qt).view(sz_b, len_qt, len_v, -1, n_head, d_k)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.permute(0, 4, 1, 2, 3, 5), k.permute(0, 4, 1, 2, 3, 5), v.permute(0, 4, 1, 2, 3, 5)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        output, attn = self.attention.interpolate(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        output = output.transpose(1, 2).contiguous().view(sz_b, len_qt, -1)
        output = self.dropout(self.fc(output))
        output += residual

        if not self.normalize_before:
            output = self.layer_norm(output)

        return output, attn


class PositionwiseFeedForward(nn.Module):
    """ Two-layer position-wise feed-forward neural network. """

    def __init__(self, d_in, d_hid, dropout=0.1, normalize_before=True):
        super().__init__()

        self.normalize_before = normalize_before

        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)

        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        if self.normalize_before:
            x = self.layer_norm(x)

        x = F.gelu(self.w_1(x))
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)
        x = x + residual

        if not self.normalize_before:
            x = self.layer_norm(x)
        return x


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature, attn_dropout=0.2):
        super().__init__()

        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        # if q is ODELinear, attn = (q.transpose(2, 3).flip(dims=[-2]) / self.temperature * k).sum(dim=-1).sum(dim=-1)
        attn = (q / self.temperature * k).sum(dim=-1).sum(dim=-1)

        if mask is not None:
            attn = attn.masked_fill(mask, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))

        output = (attn.unsqueeze(-1) * v.sum(dim=-2)).sum(dim=-2)
        return output, attn

    def interpolate(self, q, k, v, mask=None):
        attn = (q / self.temperature * k).sum(dim=-1).sum(dim=-1)

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = F.softmax(attn, dim=-1)

        output = (attn.unsqueeze(-1) * v.sum(dim=-2)).sum(dim=-2)
        return output, attn


class Encoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(
            self,
            d_model, d_inner,
            n_layers, n_head, d_k, d_v, dropout, args, add_pe=False, normalize_before=False):
        super().__init__()

        self.d_model = d_model
        self.pe = PositionalEncoder(d_model, "cuda:0")
        self.add_pe = add_pe

        # position vector, used for temporal encoding
        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / d_model) for i in range(d_model)])

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, normalize_before=normalize_before, args=args)
            for _ in range(n_layers)])

    def temporal_enc(self, time):
        """
        Input: batch*seq_len.
        Output: batch*seq_len*d_model.
        """

        result = time.unsqueeze(-1) / self.position_vec.to(time.device)
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result

    def forward(self, src, t, slf_attn_mask=None):
        """ Encode event sequences via masked self-attention. """
        tem_enc = self.temporal_enc(t)
        if self.add_pe:
            src = self.pe(src)

        for enc_layer in self.layer_stack:
            src += tem_enc
            src, _ = enc_layer(
                src, t,
                slf_attn_mask=slf_attn_mask)
        return src

    def interpolate(self, src, t, qt, slf_attn_mask=None):
        tem_enc = self.temporal_enc(t)
        if self.add_pe:
            src = self.pe(src)

        flag = True

        for enc_layer in self.layer_stack:
            src += tem_enc
            if flag:
                src, _ = enc_layer.interpolate(
                    src, t, qt,
                    slf_attn_mask=slf_attn_mask)
                flag = False
            else:
                src, _ = enc_layer(
                    src, qt,
                    slf_attn_mask=slf_attn_mask)
        return src

class EncoderLayer(nn.Module):
    """ Compose with two layers """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, normalize_before=True, args=None):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout, normalize_before=normalize_before, args_ode=args)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout, normalize_before=normalize_before)

    def forward(self, enc_input, time_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, time_input, mask=slf_attn_mask)

        enc_output = self.pos_ffn(enc_output)

        return enc_output, enc_slf_attn

    def interpolate(self, enc_input, time_input, query_time_point, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn.interpolate(
            enc_input, enc_input, enc_input, time_input, query_time_point, mask=slf_attn_mask)

        enc_output = self.pos_ffn(enc_output)

        return enc_output, enc_slf_attn



class Model(nn.Module):
    def __init__(self, configs):

        d_model = configs.d_model
        args_ode = {
            'use_ode': True, 'actfn': 'tanh', 'layer_type': 'concat', 'zero_init': False,
            'atol': 1e-1, 'rtol': 1e-1, 'method': 'rk4', 'regularize': False,
            'approximate_method': 'linear', 'nlinspace': 1, 'linear_type': 'before',
            'interpolate': 'cubic', 'itol': 1e-2
        }
        super().__init__()
        args_ode = AttrDict(args_ode)
        self.mode = configs.mode
        stride = int(configs.patch_stride.split(',')[1])
        self.patch_len = int(configs.patch_stride.split(',')[0])
        padding = stride

        self.patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, stride, padding, configs.dropout)

        self.encoder = Encoder(
            d_model=d_model,
            d_inner=d_model * 4,
            n_layers=1,
            n_head=4,
            d_k=d_model // 4,
            d_v=d_model // 4,
            dropout=0.1,
            args=args_ode,
            add_pe=False,
            normalize_before=False
        )

        self.pred_len = configs.pred_len
        assert self.pred_len % self.patch_len == 0
        self.lin_input = nn.Linear(configs.enc_in, d_model)

        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, self.patch_len),
        )

    def padding_forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)
        last_timestamp = enc_out[:, -1:, :].repeat(1, self.pred_len // self.patch_len, 1)
        src = torch.cat((enc_out, last_timestamp), dim=1)

        time_steps_to_predict = torch.linspace(0, 1, src.shape[1]).to(x_enc.device)
        time_steps_to_predict = time_steps_to_predict.unsqueeze(0).repeat(src.shape[0], 1)

        latents = self.encoder(src, time_steps_to_predict)
        outputs = self.decoder(latents)

        dec_out = outputs[:, -self.pred_len // self.patch_len:, :]  # [bs*n_vars, num_patch, patch_len]
        dec_out = dec_out.reshape(-1, n_vars, self.pred_len).transpose(-1, -2)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out


    def interpolate_forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        qt = torch.linspace(0, 1, enc_out.shape[1] + self.pred_len // self.patch_len).to(x_enc.device)
        qt = qt.unsqueeze(0).repeat(enc_out.shape[0], 1)
        t = qt[:, :enc_out.shape[1]]

        latents = self.encoder.interpolate(enc_out, t, qt)
        outputs = self.decoder(latents)  # [bs*n_var, patch_len, num_patch]

        dec_out = outputs[:, -self.pred_len // self.patch_len:, :]  # [bs*n_vars, num_patch, patch_len]
        dec_out = dec_out.reshape(-1, n_vars, self.pred_len).transpose(-1, -2)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        if self.mode == 'padding':
            return self.padding_forward(x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None)
        else:
            return self.interpolate_forward(x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None)
