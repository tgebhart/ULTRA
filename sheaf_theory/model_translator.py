from itertools import product
import torch
from torch_geometric.utils import get_laplacian, to_dense_adj, add_self_loops, to_scipy_sparse_matrix
import numpy as np
from scipy.sparse.linalg import eigs, eigsh
from tqdm import tqdm
from sheaf_theory import batch_harmonic_extension as bhe


def coboundary(edge_index,Fh,Ft,relabel=False):
    device = Fh.device
    if relabel:
        _, edge_index = torch.unique(edge_index, sorted=True, return_inverse=True)
    ne = edge_index.shape[-1]
    nv = edge_index.max() + 1
    de = Fh.shape[-2]
    dv = Fh.shape[-1]
    idxs = []
    vals = torch.zeros(0, device=device)
    for e in range(ne):
        h = edge_index[0,e]
        t = edge_index[1,e]
        r = list(range(e*de,(e+1)*de))
        idxs += list(product(r, list(range(h*dv,(h+1)*dv)))) + \
              list(product(r, list(range(t*dv,(t+1)*dv))))
        vals = torch.cat((vals, Fh[e,:,:].flatten(), -Ft[e,:,:].flatten()))
    return torch.sparse_coo_tensor(torch.LongTensor(idxs).T, vals, size=(ne*de,nv*dv), device=device)

def translate_to_graph_rep_inv(model, data, normalization=None, copy_weights=True,
                               self_loops=True, atol=None, threshold_pctile=0):
    nbf_layers = model.layers
    if copy_weights:
        nbf_layers = [nbf_layers[0]]
    for layer in nbf_layers:
        if hasattr(layer, 'relation_linear'):
            rels = layer.relation_linear(data).weight
        elif hasattr(layer, 'relation'):
            rels = layer.relation.weight
        else:
            raise ValueError('Cannot find relation representation in layer.')

        # now we're assuming the relation representation is a vector, implying 
        # the associated sheaf Laplacian is the direct sum of the Laplacian in 
        # each dimension
        rels_reshaped = torch.index_select(rels, 0, data.edge_type)

        L_edge_index, L_edge_weight = get_laplacian(data.edge_index, edge_weight=rels_reshaped, normalization=normalization)
        L = to_dense_adj(L_edge_index, edge_attr=L_edge_weight, max_num_nodes=data.num_nodes)
        L = L[0] # take first (and only) in batch
        # Linv = L
        # for i in range(2,100):
        #     Linv += torch.linalg.matrix_power(I - L, i)[0]
        # print('computing inverse...')

        res = []
        for d in tqdm(range(rels_reshaped.shape[1])):
            Linv = torch.linalg.pinv(L[:,:,d], atol=atol)
            res.append(Linv.unsqueeze(-1).to('cpu')) # store on cpu to save memory

        res = torch.concatenate(res, axis=-1)
        res = res.to(rels.device)
        # I = torch.eye(data.num_nodes).to(rels.device)
        # I = I.unsqueeze(-1)
        # res = res - I

        if threshold_pctile > 0:
            # Reshape the tensor to collapse the first two dimensions
            reshaped_tensor = res.view(-1, res.size(-1))

            # Calculate the 95th percentile along the new dimension (0)
            percentiles = torch.quantile(reshaped_tensor, threshold_pctile, dim=0, keepdim=True)

            # Reshape percentiles to enable broadcasting (1, 1, number of channels)
            percentiles = percentiles.view(1, 1, res.size(-1))

            # Threshold the tensor: values below the 95th percentile are set to zero
            res = torch.where(res >= percentiles, res, torch.tensor(0.0))
            # percentiles = torch.quantile(res.to('cpu'), threshold_pctile, dim=None)
            # res = torch.where(res >= percentiles.to(res.device), res, torch.tensor(0.0))

        return res



def translate_to_graph_rep_eig(model, data, k=32, normalization=None, copy_weights=True,
                               self_loops=True, atol=1e-6):
    nbf_layers = model.layers
    if copy_weights:
        nbf_layers = [nbf_layers[0]]
    for layer in nbf_layers:
        if hasattr(layer, 'relation_linear'):
            rels = layer.relation_linear(data).weight
        elif hasattr(layer, 'relation'):
            rels = layer.relation.weight
        else:
            raise ValueError('Cannot find relation representation in layer.')

        # now we're assuming the relation representation is a vector, implying 
        # the associated sheaf Laplacian is the direct sum of the Laplacian in 
        # each dimension
        rels_reshaped = torch.index_select(rels, 0, data.edge_type)
        I = torch.eye(data.num_nodes).to(rels.device)
        I = I.unsqueeze(-1).repeat(1,1,rels_reshaped.shape[1])

        L_edge_index, L_edge_weight = get_laplacian(data.edge_index, edge_weight=rels_reshaped, normalization=normalization)

        res = []
        for d in range(rels_reshaped.shape[1]): 
            
            Ld = to_scipy_sparse_matrix(L_edge_index, edge_attr=L_edge_weight[:, d], num_nodes=data.num_nodes)
            eig_fn = eigs # if not self.is_undirected else eigsh

            eig_vals, eig_vecs = eig_fn(  # type: ignore
                Ld,
                k=k,
                which='SR',
                return_eigenvectors=True,
            )
            eig_sort = eig_vals.argsort()[::-1]
            eig_vecs = np.real(eig_vecs[:, eig_sort])
            
            eig_vals = torch.from_numpy(np.real(eig_vals[eig_sort]))
            eig_vals[eig_vals > atol] = 1/eig_vals[eig_vals > atol]
            eig_vals[eig_vals < atol] = 0
            pe  = torch.from_numpy(eig_vecs[:, :k])
            pe = pe * eig_vals @ pe.T

            res.append(pe.unsqueeze(-1))
        res = torch.concatenate(res, axis=-1).to(rels.device)

        return res