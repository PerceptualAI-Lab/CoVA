from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)

def compute_metrics(x):
    if x is None:
        logger.error("compute_metrics: Input is None")
        return {
            'R1': 0, 'R5': 0, 'R10': 0, 
            'MR': 1, 'MedianR': 1, 'MeanR': 1, 
            'cols': []
        }
    
    if x.ndim != 2:
        logger.error(f"compute_metrics: Expected 2D array, got shape {x.shape}")
        return {
            'R1': 0, 'R5': 0, 'R10': 0,
            'MR': 1, 'MedianR': 1, 'MeanR': 1,
            'cols': []
        }
    
    if x.shape[0] == 0 or x.shape[1] == 0:
        logger.error(f"compute_metrics: Empty matrix with shape {x.shape}")
        return {
            'R1': 0, 'R5': 0, 'R10': 0,
            'MR': 1, 'MedianR': 1, 'MeanR': 1,
            'cols': []
        }
    
    nan_count = np.isnan(x).sum()
    inf_count = np.isinf(x).sum()
    
    if nan_count > 0 or inf_count > 0:
        logger.warning(f"compute_metrics: Found {nan_count} NaNs and {inf_count} Infs in matrix")
        logger.warning(f"Matrix stats before cleaning: min={np.nanmin(x):.4f}, max={np.nanmax(x):.4f}, mean={np.nanmean(x):.4f}")
        
        x = np.nan_to_num(x, nan=0.0, posinf=100.0, neginf=-100.0)
        
        logger.warning(f"Matrix stats after cleaning: min={x.min():.4f}, max={x.max():.4f}, mean={x.mean():.4f}")
    
    try:
        sx = np.sort(-x, axis=1)
        d = np.diag(-x)
        d = d[:, np.newaxis]
        ind = sx - d
        ind = np.where(ind == 0)
        
        if len(ind) < 2:
            logger.error("compute_metrics: Invalid ind structure")
            return {
                'R1': 0, 'R5': 0, 'R10': 0,
                'MR': 1, 'MedianR': 1, 'MeanR': 1,
                'cols': []
            }
        
        ind = ind[1]
        
        if len(ind) == 0:
            logger.error("compute_metrics: No matching indices found (ind is empty)")
            logger.error(f"Matrix diagonal: {np.diag(x)}")
            logger.error(f"Matrix shape: {x.shape}")
            
            for i in range(min(5, x.shape[0])):
                logger.error(f"Row {i}: min={x[i].min():.4f}, max={x[i].max():.4f}, diag={x[i,i]:.4f}")
            
            return {
                'R1': 0, 'R5': 0, 'R10': 0,
                'MR': 1, 'MedianR': 1, 'MeanR': 1,
                'cols': []
            }
        
        metrics = {}
        metrics['R1'] = float(np.sum(ind == 0)) * 100 / len(ind)
        metrics['R5'] = float(np.sum(ind < 5)) * 100 / len(ind)
        metrics['R10'] = float(np.sum(ind < 10)) * 100 / len(ind)
        metrics['MR'] = np.median(ind) + 1
        metrics["MedianR"] = metrics['MR']
        metrics["MeanR"] = np.mean(ind) + 1
        metrics["cols"] = [int(i) for i in list(ind)]
        
        logger.info(f"compute_metrics successful: R@1={metrics['R1']:.2f}, R@5={metrics['R5']:.2f}, R@10={metrics['R10']:.2f}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"compute_metrics: Exception occurred: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        return {
            'R1': 0, 'R5': 0, 'R10': 0,
            'MR': 1, 'MedianR': 1, 'MeanR': 1,
            'cols': []
        }

def print_computed_metrics(metrics):
    r1 = metrics['R1']
    r5 = metrics['R5']
    r10 = metrics['R10']
    mr = metrics['MR']
    print('R@1: {:.4f} - R@5: {:.4f} - R@10: {:.4f} - Median R: {}'.format(r1, r5, r10, mr))

def tensor_text_to_video_metrics(sim_tensor, top_k=[1,5,10]):
    if not torch.is_tensor(sim_tensor):
        sim_tensor = torch.tensor(sim_tensor)
    
    stacked_sim_matrices = sim_tensor.permute(1, 0, 2)
    first_argsort = torch.argsort(stacked_sim_matrices, dim=-1, descending=True)
    second_argsort = torch.argsort(first_argsort, dim=-1, descending=False)
    
    ranks = torch.flatten(torch.diagonal(second_argsort, dim1=1, dim2=2))
    
    permuted_original_data = torch.flatten(torch.diagonal(sim_tensor, dim1=0, dim2=2))
    mask = ~torch.logical_or(torch.isinf(permuted_original_data), torch.isnan(permuted_original_data))
    valid_ranks = ranks[mask]
    
    if not torch.is_tensor(valid_ranks):
        valid_ranks = torch.tensor(valid_ranks)
    
    results = {f"R{k}": float(torch.sum(valid_ranks < k) * 100 / len(valid_ranks)) for k in top_k}
    results["MedianR"] = float(torch.median(valid_ranks + 1))
    results["MeanR"] = float(np.mean(valid_ranks.numpy() + 1))
    results["Std_Rank"] = float(np.std(valid_ranks.numpy() + 1))
    results['MR'] = results["MedianR"]
    return results

def tensor_video_to_text_sim(sim_tensor):
    if not torch.is_tensor(sim_tensor):
        sim_tensor = torch.tensor(sim_tensor)
    
    sim_tensor[sim_tensor != sim_tensor] = float('-inf')
    values, _ = torch.max(sim_tensor, dim=1, keepdim=True)
    return torch.squeeze(values).T

def compute_metrics_rect(sim, gt_cols=None, ks=(1,5,10)):

    import numpy as np
    if sim is None or getattr(sim, "ndim", 0) != 2 or sim.shape[0] == 0 or sim.shape[1] == 0:
        return {"R1":0,"R5":0,"R10":0,"MR":1,"MedianR":1,"MeanR":1}

    sim = np.asarray(sim, dtype=np.float32)

    n_nan, n_inf = np.isnan(sim).sum(), np.isinf(sim).sum()
    if n_nan or n_inf:
        logger.warning(f"[metrics] sim has NaN={n_nan}, Inf={n_inf}. Cleaning...")
        sim = np.nan_to_num(sim, nan=0.0, posinf=100.0, neginf=-100.0)

    Nq, Nc = sim.shape

    if gt_cols is None:
        Nt = min(Nq, Nc)
        gt_cols = np.arange(Nq, dtype=np.int64)
        gt_cols = np.clip(gt_cols, 0, Nt-1)

    gt_cols = np.asarray(gt_cols, dtype=np.int64)
    if gt_cols.shape[0] != Nq:
        raise ValueError(f"[metrics] gt_cols length {gt_cols.shape[0]} != Nq {Nq}")
    if (gt_cols < 0).any() or (gt_cols >= Nc).any():
        raise ValueError("[metrics] gt_cols has out-of-range indices")


    ranks = np.empty(Nq, dtype=np.int32)
    for i in range(Nq):
        order = np.argsort(-sim[i], axis=-1)
        ranks[i] = int(np.where(order == gt_cols[i])[0][0]) + 1

    out = {f"R{k}": float(np.mean(ranks <= k) * 100.0) for k in ks}
    out["MR"] = float(np.median(ranks))
    out["MedianR"] = out["MR"]
    out["MeanR"]  = float(np.mean(ranks))
    return out
