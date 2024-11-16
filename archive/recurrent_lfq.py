# type: ignore
# flake8: noqa

# This idea hasn't worked yet but it's interesting!

"""
Lookup Free Quantization
Proposed in https://arxiv.org/abs/2310.05737

In the simplest setup, each dimension is quantized into {-1, 1}.
An entropy penalty is used to encourage utilization.
"""

from collections import namedtuple
from contextlib import nullcontext
from functools import cache, partial
from math import ceil, log2

import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import pack, rearrange, reduce, unpack
from torch import einsum, nn
from torch.amp import autocast
from torch.distributed import nn as dist_nn
from torch.nn import Module

# constants

Return = namedtuple("Return", ["quantized", "indices", "entropy_aux_loss"])

LossBreakdown = namedtuple(
    "LossBreakdown", ["per_sample_entropy", "batch_entropy", "commitment"]
)

# distributed helpers


@cache
def is_distributed():
    return dist.is_initialized() and dist.get_world_size() > 1


def maybe_distributed_mean(t):
    if not is_distributed():
        return t

    dist_nn.all_reduce(t)
    t = t / dist.get_world_size()
    return t


# helper functions


def exists(v):
    return v is not None


def identity(t):
    return t


def default(*args):
    for arg in args:
        if exists(arg):
            return arg() if callable(arg) else arg
    return None


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


def l2norm(t):
    return F.normalize(t, dim=-1)


# entropy


def log(t, eps=1e-5):
    return t.clamp(min=eps).log()


def entropy(prob):
    return (-prob * log(prob)).sum(dim=-1)


# cosine sim linear


class CosineSimLinear(Module):
    def __init__(self, dim_in, dim_out, scale=1.0):
        super().__init__()
        self.scale = scale
        self.weight = nn.Parameter(torch.randn(dim_in, dim_out))

    def forward(self, x):
        x = F.normalize(x, dim=-1)
        w = F.normalize(self.weight, dim=0)
        return (x @ w) * self.scale


class RecurrentBinaryQuantizer(nn.Module):
    def __init__(self, n_outputs, d_embed, codebook_value=1):
        super().__init__()
        self.n_outputs = n_outputs
        self.gru = nn.GRU(1, d_embed, batch_first=True)
        self.output_layer = nn.Linear(d_embed, 1)
        self.codebook_value = codebook_value

    def forward(self, embedding):
        B, L, _ = embedding.shape
        BL = B * L
        embedding = embedding.view(1, BL, -1)

        hidden_state = torch.zeros(1, BL,
                                   embedding.size(-1), device=embedding.device)
        input_step = torch.zeros(BL, 1, 1, device=embedding.device)

        outputs = torch.zeros(BL, self.n_outputs, device=embedding.device)
        quantized = torch.zeros(BL, self.n_outputs, device=embedding.device)
        for idx in range(self.n_outputs):
            gru_output, hidden_state = self.gru(input_step,
                                                hidden_state + embedding)
            gru_output = self.output_layer(gru_output)
            outputs[:, idx] = gru_output.squeeze()

            q = torch.where(gru_output > 0,
                            self.codebook_value, -self.codebook_value)
            quantized[:, idx] = q.squeeze()
            input_step = q.float()

        return outputs.view(B, L, -1), quantized.view(B, L, -1)


# class


class RecurrentLFQ(Module):
    def __init__(
        self,
        *,
        dim=None,
        codebook_size=None,
        d_embed=None,
        entropy_loss_weight=0.1,
        commitment_loss_weight=0.0,
        diversity_gamma=1.0,
        straight_through_activation=nn.Identity(),
        num_codebooks=1,
        keep_num_codebooks_dim=None,
        codebook_scale=1.0,  # for residual LFQ, codebook scaled down by 2x at each layer
        frac_per_sample_entropy=1.0,  # make less than 1. to only use a random fraction of the probs for per sample entropy
        has_projections=None,
        projection_has_bias=True,
        soft_clamp_input_value=None,
        cosine_sim_project_in=False,
        cosine_sim_project_in_scale=None,
        channel_first=None,
        experimental_softplus_entropy_loss=False,
        entropy_loss_offset=5.0,  # how much to shift the loss before softplus
        spherical=False,  # from https://arxiv.org/abs/2406.07548
        force_quantization_f32=True,  # will force the quantization step to be full precision
    ):
        super().__init__()

        # some assert validations

        assert exists(dim) or exists(
            codebook_size
        ), "either dim or codebook_size must be specified for LFQ"
        assert exists(d_embed), "d_embed"
        assert (
            not exists(codebook_size) or log2(codebook_size).is_integer()
        ), f"your codebook size must be a power of 2 for lookup free quantization (suggested {2 ** ceil(log2(codebook_size))})"

        codebook_size = default(codebook_size, lambda: 2**dim)
        self.codebook_size = codebook_size

        codebook_dim = int(log2(codebook_size))
        codebook_dims = codebook_dim * num_codebooks
        dim = default(dim, codebook_dims)
        self.d_embed = d_embed

        has_projections = default(has_projections, dim != codebook_dims)

        if cosine_sim_project_in:
            cosine_sim_project_in = default(cosine_sim_project_in_scale, codebook_scale)
            project_in_klass = partial(CosineSimLinear, scale=cosine_sim_project_in)
        else:
            project_in_klass = partial(nn.Linear, bias=projection_has_bias)

        self.project_in = (
            project_in_klass(dim, codebook_dims) if has_projections else nn.Identity()
        )
        self.project_out = (
            nn.Linear(codebook_dims, dim, bias=projection_has_bias)
            if has_projections
            else nn.Identity()
        )
        self.has_projections = has_projections

        self.dim = dim
        self.codebook_dim = codebook_dim
        self.num_codebooks = num_codebooks

        keep_num_codebooks_dim = default(keep_num_codebooks_dim, num_codebooks > 1)
        assert not (num_codebooks > 1 and not keep_num_codebooks_dim)
        self.keep_num_codebooks_dim = keep_num_codebooks_dim

        # channel first

        self.channel_first = channel_first

        # straight through activation

        self.activation = straight_through_activation

        # whether to use BSQ (binary spherical quantization)

        self.spherical = spherical
        self.maybe_l2norm = (
            (lambda t: l2norm(t) * self.codebook_scale) if spherical else identity
        )

        # entropy aux loss related weights

        assert 0 < frac_per_sample_entropy <= 1.0
        self.frac_per_sample_entropy = frac_per_sample_entropy

        self.diversity_gamma = diversity_gamma
        self.entropy_loss_weight = entropy_loss_weight

        # codebook scale

        self.codebook_scale = codebook_scale

        # commitment loss

        self.commitment_loss_weight = commitment_loss_weight

        # whether to soft clamp the input value from -value to value

        self.soft_clamp_input_value = soft_clamp_input_value
        assert (
            not exists(soft_clamp_input_value)
            or soft_clamp_input_value >= codebook_scale
        )

        # whether to make the entropy loss positive through a softplus (experimental, please report if this worked or not in discussions)

        self.entropy_loss_offset = entropy_loss_offset
        self.experimental_softplus_entropy_loss = experimental_softplus_entropy_loss

        # for no auxiliary loss, during inference

        self.register_buffer("mask", 2 ** torch.arange(codebook_dim - 1, -1, -1))
        self.register_buffer("zero", torch.tensor(0.0), persistent=False)

        # whether to force quantization step to be f32

        self.force_quantization_f32 = force_quantization_f32

        # codes

        all_codes = torch.arange(codebook_size)
        bits = ((all_codes[..., None].int() & self.mask) != 0).float()
        codebook = self.bits_to_codes(bits)

        self.rbq = RecurrentBinaryQuantizer(
            int(log2(codebook_size)), d_embed, codebook_scale
        )

        self.register_buffer("codebook", codebook.float(), persistent=False)

    def bits_to_codes(self, bits):
        return bits * self.codebook_scale * 2 - self.codebook_scale

    @property
    def dtype(self):
        return self.codebook.dtype

    def indices_to_codes(self, indices, project_out=True):
        is_img_or_video = indices.ndim >= (3 + int(self.keep_num_codebooks_dim))
        should_transpose = default(self.channel_first, is_img_or_video)

        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, "... -> ... 1")

        # indices to codes, which are bits of either -1 or 1

        bits = ((indices[..., None].int() & self.mask) != 0).to(self.dtype)

        codes = self.bits_to_codes(bits)

        codes = self.maybe_l2norm(codes)

        codes = rearrange(codes, "... c d -> ... (c d)")

        # whether to project codes out to original dimensions
        # if the input feature dimensions were not log2(codebook size)

        if project_out:
            codes = self.project_out(codes)

        # rearrange codes back to original shape

        if should_transpose:
            codes = rearrange(codes, "b ... d -> b d ...")

        return codes

    def forward(
        self,
        x,
        inv_temperature=100.0,
        return_loss_breakdown=False,
        mask=None,
    ):
        """
        einstein notation
        b - batch
        n - sequence (or flattened spatial dimensions)
        d - feature dimension, which is also log2(codebook size)
        c - number of codebook dim
        """

        is_img_or_video = x.ndim >= 4
        should_transpose = default(self.channel_first, is_img_or_video)

        # standardize image or video into (batch, seq, dimension)

        if should_transpose:
            x = rearrange(x, "b d ... -> b ... d")
            x, ps = pack_one(x, "b * d")
        # CHANGE IN RECURRENT LFQ
        assert (
            x.shape[-1] == self.d_embed
        ), f"expected dimension of {self.d_embed} but received {x.shape[-1]}"

        # ------------------------ RECURRENT LFQ EDITS ------------------------

        x, quantized = self.rbq(x)
        x = rearrange(x, "b n (c d) -> b n c d", c=self.num_codebooks)
        quantized = rearrange(quantized, "b n (c d) -> b n c d", c=self.num_codebooks)

        # whether to force quantization step to be full precision or not

        force_f32 = self.force_quantization_f32

        quantization_context = (
            partial(autocast, "cuda", enabled=False) if force_f32 else nullcontext
        )

        with quantization_context():

            if force_f32:
                orig_dtype = x.dtype
                x = x.float()

            # quantize by eq 3.

            original_input = x

            # codebook_value = torch.ones_like(x) * self.codebook_scale
            # quantized = torch.where(x > 0, codebook_value, -codebook_value)

            # calculate indices

            indices = reduce(
                (quantized > 0).int() * self.mask.int(), "b n c d -> b n c", "sum"
            )

            # maybe l2norm

            quantized = self.maybe_l2norm(quantized)

            # use straight-through gradients (optionally with custom activation fn) if training

            if self.training:
                x = self.activation(x)
                x = x + (quantized - x).detach()
            else:
                x = quantized

            # entropy aux loss

            if self.training:

                if force_f32:
                    codebook = self.codebook.float()

                codebook = self.maybe_l2norm(codebook)

                # the same as euclidean distance up to a constant
                distance = -2 * einsum(
                    "... i d, j d -> ... i j", original_input, codebook
                )

                prob = (-distance * inv_temperature).softmax(dim=-1)

                # account for mask

                if exists(mask):
                    prob = prob[mask]
                else:
                    prob = rearrange(prob, "b n ... -> (b n) ...")

                # whether to only use a fraction of probs, for reducing memory

                if self.frac_per_sample_entropy < 1.0:
                    num_tokens = prob.shape[0]
                    num_sampled_tokens = int(num_tokens * self.frac_per_sample_entropy)
                    rand_mask = (
                        torch.randn(num_tokens).argsort(dim=-1) < num_sampled_tokens
                    )
                    per_sample_probs = prob[rand_mask]
                else:
                    per_sample_probs = prob

                # calculate per sample entropy

                per_sample_entropy = entropy(per_sample_probs).mean()

                # distribution over all available tokens in the batch

                avg_prob = reduce(per_sample_probs, "... c d -> c d", "mean")

                avg_prob = maybe_distributed_mean(avg_prob)

                codebook_entropy = entropy(avg_prob).mean()

                # 1. entropy will be nudged to be low for each code, to encourage the network to output confident predictions
                # 2. codebook entropy will be nudged to be high, to encourage all codes to be uniformly used within the batch

                entropy_aux_loss = (
                    per_sample_entropy - self.diversity_gamma * codebook_entropy
                )
            else:
                # if not training, just return dummy 0
                entropy_aux_loss = per_sample_entropy = codebook_entropy = self.zero

            # whether to make the entropy loss positive or not through a (shifted) softplus

            if self.training and self.experimental_softplus_entropy_loss:
                entropy_aux_loss = F.softplus(
                    entropy_aux_loss + self.entropy_loss_offset
                )

            # commit loss

            if self.training and self.commitment_loss_weight > 0.0:

                commit_loss = F.mse_loss(
                    original_input, quantized.detach(), reduction="none"
                )

                if exists(mask):
                    commit_loss = commit_loss[mask]

                commit_loss = commit_loss.mean()
            else:
                commit_loss = self.zero

            # input back to original dtype if needed

            if force_f32:
                x = x.type(orig_dtype)

        # merge back codebook dim

        x = rearrange(x, "b n c d -> b n (c d)")

        # project out to feature dimension if needed

        x = self.project_out(x)

        # reconstitute image or video dimensions

        if should_transpose:
            x = unpack_one(x, ps, "b * d")
            x = rearrange(x, "b ... d -> b d ...")

            indices = unpack_one(indices, ps, "b * c")

        # whether to remove single codebook dim

        if not self.keep_num_codebooks_dim:
            indices = rearrange(indices, "... 1 -> ...")

        # complete aux loss

        aux_loss = (
            entropy_aux_loss * self.entropy_loss_weight
            + commit_loss * self.commitment_loss_weight
        )

        # returns

        ret = Return(x, indices, aux_loss)

        if not return_loss_breakdown:
            return ret

        return ret, LossBreakdown(per_sample_entropy, codebook_entropy, commit_loss)