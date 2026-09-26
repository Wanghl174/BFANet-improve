import torch
from torch.autograd import Function
import BFANet_lib


def density_normalized_soft_boundary(xyz, sem, batch_offset, k=16, tau=1.0,
                                     eps=1e-6, chunk_size=2048):
    """Compute density-normalized soft boundary targets.

    For each point, the median distance to its K nearest (same-scene)
    neighbours defines the local scale ``s_i``.  Distances are normalized by
    ``sqrt(s_i*s_j)`` and converted to Gaussian weights before accumulating
    the weighted fraction of neighbours with a different semantic label.
    """
    if xyz.numel() == 0:
        return xyz.new_empty((0,), dtype=torch.float32)
    out = []
    starts = [0] + batch_offset.detach().cpu().tolist()
    for b in range(len(starts) - 1):
        lo, hi = int(starts[b]), int(starts[b + 1])
        pts, labels = xyz[lo:hi], sem[lo:hi]
        n = pts.shape[0]
        if n <= 1:
            out.append(pts.new_zeros((n,), dtype=torch.float32))
            continue
        kk = min(max(int(k), 1), n - 1)
        # cdist is chunked to avoid materializing a large scene distance matrix.
        scales = []
        knn_indices = []
        for q0 in range(0, n, chunk_size):
            d = torch.cdist(pts[q0:q0 + chunk_size], pts)
            rows = torch.arange(d.shape[0], device=d.device)
            d[rows, q0 + rows] = float('inf')
            vals, idx = torch.topk(d, kk, dim=1, largest=False)
            scales.append(vals.median(dim=1).values)
            knn_indices.append((vals, idx))
        s = torch.cat(scales).clamp_min(eps)
        q_parts = []
        for q0, (vals, idx) in zip(range(0, n, chunk_size), knn_indices):
            sj = s[idx]
            delta = vals / torch.sqrt(s[q0:q0 + vals.shape[0], None] * sj + eps)
            weights = torch.exp(-(delta.square()) / (2.0 * float(tau) ** 2))
            different = (labels[idx] != labels[q0:q0 + vals.shape[0], None])
            different &= (labels[idx] >= 0) & (labels[q0:q0 + vals.shape[0], None] >= 0)
            q_parts.append((weights * different.to(weights.dtype)).sum(1) /
                           (weights.sum(1) + eps))
        out.append(torch.cat(q_parts).to(torch.float32))
    return torch.cat(out, dim=0)
class Cluster(Function):
    @staticmethod
    def forward(ctx, xyz_cc, cc_bp, sem, radius, batch_size):
        # #### offseted xyz
        x = xyz_cc[:, 0].type(torch.float32).contiguous()
        y = xyz_cc[:, 1].type(torch.float32).contiguous()
        z = xyz_cc[:, 2].type(torch.float32).contiguous()
        l1_norm = torch.abs(x) + torch.abs(y) + torch.abs(z)
        index_mapper_list = []
        for batch_i in range(batch_size):
            num_bp = cc_bp[batch_i]
            index_mapper = torch.arange(0, num_bp, 1)
            index_mapper_list.append(index_mapper)
        index_mapper = torch.cat(index_mapper_list, dim=0).type(torch.int32).contiguous()

        # ####sem pred
        sem = sem.type(torch.int32).contiguous()

        # ####points nums and cluster id init
        ins_point_num = xyz_cc.shape[0]
        cluster_id = torch.ones(ins_point_num) * -1
        cluster_id = cluster_id.type(torch.int32).contiguous()
        # ####clusters nums for each batch
        batch_size = cc_bp.shape[0]
        cluster_num = torch.zeros([batch_size]).type(torch.int32).contiguous()
        # ####init cluster semantic address
        clt_sem = torch.zeros(ins_point_num, dtype=torch.int32).contiguous()

        # ####double check contiguous for pointer(*)
        assert x.is_contiguous()
        assert y.is_contiguous()
        assert z.is_contiguous()
        assert l1_norm.is_contiguous()
        assert index_mapper.is_contiguous()
        assert sem.is_contiguous()
        assert cc_bp.is_contiguous()
        assert cluster_id.is_contiguous()
        assert clt_sem.is_contiguous()

        BFANet_lib.bfanet_cluster(x, y, z, l1_norm, index_mapper, cc_bp, sem, cluster_id, cluster_num, clt_sem, batch_size, radius)
        return cluster_id, cluster_num, clt_sem

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None, None


cluster = Cluster.apply


class Detect_margin(Function):
    @staticmethod
    def forward(ctx, xyz_original, sem, batch_offset, radius):

        # ####original xyz
        xo = xyz_original[:, 0].type(torch.float32).contiguous()
        yo = xyz_original[:, 1].type(torch.float32).contiguous()
        zo = xyz_original[:, 2].type(torch.float32).contiguous()
        # ####semantic pred
        sem = sem.type(torch.int32).contiguous()

        # ####clusters nums for each batch
        batch_size = batch_offset.shape[0]
        batch_offset = batch_offset.type(torch.int32).contiguous()

        # ####margin queue
        margin_queue = torch.zeros(xo.shape[0], dtype=torch.int32).contiguous()


        # ####double check contiguous for pointer(*)
        assert xo.is_contiguous()
        assert yo.is_contiguous()
        assert zo.is_contiguous()
        assert sem.is_contiguous()
        assert margin_queue.is_contiguous()
        assert batch_offset.is_contiguous()

        BFANet_lib.sem_margin_det(xo, yo, zo, sem, batch_offset, margin_queue, radius, batch_size)
        # test = margin_queue.detach().cpu().numpy()
        return margin_queue

    @staticmethod
    def backward(ctx, a=None):
        return None


detect_margin = Detect_margin.apply



class Avg_feat(Function):
    @staticmethod
    def forward(ctx, xyz_original, sem, feat_in, batch_offset, radius):

        # ####clusters nums for each batch
        batch_size = batch_offset.shape[0]
        batch_offset = batch_offset.type(torch.int32).contiguous()

        # ####original xyz
        xo = xyz_original[:, 0].type(torch.float32).contiguous()
        yo = xyz_original[:, 1].type(torch.float32).contiguous()
        zo = xyz_original[:, 2].type(torch.float32).contiguous()

        l1_norm = torch.abs(xo) + torch.abs(yo) + torch.abs(zo)
        index_mapper_list = []
        for batch_i in range(batch_size):
            num_bp = batch_offset[batch_i]
            index_mapper = torch.arange(0, num_bp, 1)
            index_mapper_list.append(index_mapper)
        index_mapper = torch.cat(index_mapper_list, dim=0).type(torch.int32).contiguous()

        # ####semantic pred
        sem = sem.type(torch.int32).contiguous()

        # ####points nums and cluster id init
        ins_point_num = xyz_original.shape[0]
        cluster_id = torch.ones(ins_point_num) * -1
        cluster_id = cluster_id.type(torch.int32).contiguous()

        # ####feat queue
        feat_in_queue = feat_in.view(-1).type(torch.float32).contiguous()
        feat_out_queue = torch.zeros_like(feat_in_queue).type(torch.float32).contiguous()

        dim_num = feat_in.shape[-1]

        # ####double check contiguous for pointer(*)
        assert xo.is_contiguous()
        assert yo.is_contiguous()
        assert zo.is_contiguous()
        assert l1_norm.is_contiguous()
        assert index_mapper.is_contiguous()
        assert sem.is_contiguous()
        assert feat_in_queue.is_contiguous()
        assert feat_out_queue.is_contiguous()
        assert batch_offset.is_contiguous()
        assert cluster_id.is_contiguous()

        BFANet_lib.avg_feat_radius(xo, yo, zo,  batch_offset, sem, feat_in_queue, feat_out_queue, l1_norm, index_mapper, cluster_id,  batch_size, radius, dim_num)
        # test = feat_out_queue.view(-1, dim_num).detach().cpu().numpy()
        return feat_out_queue.view(-1, dim_num)

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


avg_feat = Avg_feat.apply


