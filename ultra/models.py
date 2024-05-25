import copy
import torch
from tqdm import tqdm
from torch import nn
from torch_geometric.utils import get_laplacian, to_dense_adj
from . import tasks, layers
from ultra.base_nbfnet import BaseNBFNet, index_to_mask

class Ultra(nn.Module):

    def __init__(self, rel_model_cfg, entity_model_cfg):
        # kept that because super Ultra sounds cool
        super(Ultra, self).__init__()

        self.relation_model = RelNBFNet(**rel_model_cfg)
        self.entity_model = EntityNBFNet(**entity_model_cfg)

        
    def forward(self, data, batch):
        
        # batch shape: (bs, 1+num_negs, 3)
        # relations are the same all positive and negative triples, so we can extract only one from the first triple among 1+nug_negs
        query_rels = batch[:, 0, 2]
        relation_representations = self.relation_model(data.relation_graph, query=query_rels)
        score = self.entity_model(data, relation_representations, batch)
        
        return score


# NBFNet to work on the graph of relations with 4 fundamental interactions
# Doesn't have the final projection MLP from hidden dim -> 1, returns all node representations 
# of shape [bs, num_rel, hidden]
class RelNBFNet(BaseNBFNet):

    def __init__(self, input_dim, hidden_dims, num_relation=4, **kwargs):
        super().__init__(input_dim, hidden_dims, num_relation, **kwargs)

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(
                layers.GeneralizedRelationalConv(
                    self.dims[i], self.dims[i + 1], num_relation,
                    self.dims[0], self.message_func, self.aggregate_func, self.layer_norm,
                    self.activation, dependent=False)
                )

        if self.concat_hidden:
            feature_dim = sum(hidden_dims) + input_dim
            self.mlp = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(),
                nn.Linear(feature_dim, input_dim)
            )

    
    def bellmanford(self, data, h_index, separate_grad=False):
        batch_size = len(h_index)

        # initialize initial nodes (relations of interest in the batcj) with all ones
        query = torch.ones(h_index.shape[0], self.dims[0], device=h_index.device, dtype=torch.float)
        index = h_index.unsqueeze(-1).expand_as(query)

        # initial (boundary) condition - initialize all node states as zeros
        boundary = torch.zeros(batch_size, data.num_nodes, self.dims[0], device=h_index.device)
        #boundary = torch.zeros(data.num_nodes, *query.shape, device=h_index.device)
        # Indicator function: by the scatter operation we put ones as init features of source (index) nodes
        boundary.scatter_add_(1, index.unsqueeze(1), query.unsqueeze(1))
        size = (data.num_nodes, data.num_nodes)
        edge_weight = torch.ones(data.num_edges, device=h_index.device)

        hiddens = []
        edge_weights = []
        layer_input = boundary

        for layer in self.layers:
            # Bellman-Ford iteration, we send the original boundary condition in addition to the updated node states
            hidden = layer(layer_input, query, boundary, data.edge_index, data.edge_type, size, edge_weight)
            if self.short_cut and hidden.shape == layer_input.shape:
                # residual connection here
                hidden = hidden + layer_input
            hiddens.append(hidden)
            edge_weights.append(edge_weight)
            layer_input = hidden

        # original query (relation type) embeddings
        node_query = query.unsqueeze(1).expand(-1, data.num_nodes, -1) # (batch_size, num_nodes, input_dim)
        if self.concat_hidden:
            output = torch.cat(hiddens + [node_query], dim=-1)
            output = self.mlp(output)
        else:
            output = hiddens[-1]

        return {
            "node_feature": output,
            "edge_weights": edge_weights,
        }

    def forward(self, rel_graph, query):

        # message passing and updated node representations (that are in fact relations)
        output = self.bellmanford(rel_graph, h_index=query)["node_feature"]  # (batch_size, num_nodes, hidden_dim）
        
        return output
    

class EntityNBFNet(BaseNBFNet):

    def __init__(self, input_dim, hidden_dims, num_relation=1, **kwargs):

        # dummy num_relation = 1 as we won't use it in the NBFNet layer
        super().__init__(input_dim, hidden_dims, num_relation, **kwargs)

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(
                layers.GeneralizedRelationalConv(
                    self.dims[i], self.dims[i + 1], num_relation,
                    self.dims[0], self.message_func, self.aggregate_func, self.layer_norm,
                    self.activation, dependent=False, project_relations=True)
            )

        feature_dim = (sum(hidden_dims) if self.concat_hidden else hidden_dims[-1]) + input_dim
        self.mlp = nn.Sequential()
        mlp = []
        for i in range(self.num_mlp_layers - 1):
            mlp.append(nn.Linear(feature_dim, feature_dim))
            mlp.append(nn.ReLU())
        mlp.append(nn.Linear(feature_dim, 1))
        self.mlp = nn.Sequential(*mlp)

    
    def bellmanford(self, data, h_index, r_index, separate_grad=False):
        batch_size = len(r_index)

        # initialize queries (relation types of the given triples)
        query = self.query[torch.arange(batch_size, device=r_index.device), r_index]
        index = h_index.unsqueeze(-1).expand_as(query)

        # initial (boundary) condition - initialize all node states as zeros
        boundary = torch.zeros(batch_size, data.num_nodes, self.dims[0], device=h_index.device)
        # by the scatter operation we put query (relation) embeddings as init features of source (index) nodes
        boundary.scatter_add_(1, index.unsqueeze(1), query.unsqueeze(1))
        
        size = (data.num_nodes, data.num_nodes)
        edge_weight = torch.ones(data.num_edges, device=h_index.device)

        hiddens = []
        edge_weights = []
        layer_input = boundary

        for layer in self.layers:

            # for visualization
            if separate_grad:
                edge_weight = edge_weight.clone().requires_grad_()

            # Bellman-Ford iteration, we send the original boundary condition in addition to the updated node states
            hidden = layer(layer_input, query, boundary, data.edge_index, data.edge_type, size, edge_weight)
            if self.short_cut and hidden.shape == layer_input.shape:
                # residual connection here
                hidden = hidden + layer_input
            hiddens.append(hidden)
            edge_weights.append(edge_weight)
            layer_input = hidden

        # original query (relation type) embeddings
        node_query = query.unsqueeze(1).expand(-1, data.num_nodes, -1) # (batch_size, num_nodes, input_dim)
        if self.concat_hidden:
            output = torch.cat(hiddens + [node_query], dim=-1)
        else:
            output = torch.cat([hiddens[-1], node_query], dim=-1)

        return {
            "node_feature": output,
            "edge_weights": edge_weights,
        }

    def forward(self, data, relation_representations, batch):
        h_index, t_index, r_index = batch.unbind(-1)

        # initial query representations are those from the relation graph
        self.query = relation_representations

        # initialize relations in each NBFNet layer (with uinque projection internally)
        for layer in self.layers:
            layer.relation = relation_representations

        if self.training:
            # Edge dropout in the training mode
            # here we want to remove immediate edges (head, relation, tail) from the edge_index and edge_types
            # to make NBFNet iteration learn non-trivial paths
            data = self.remove_easy_edges(data, h_index, t_index, r_index)

        shape = h_index.shape
        # turn all triples in a batch into a tail prediction mode
        h_index, t_index, r_index = self.negative_sample_to_tail(h_index, t_index, r_index, num_direct_rel=data.num_relations // 2)
        assert (h_index[:, [0]] == h_index).all()
        assert (r_index[:, [0]] == r_index).all()

        # message passing and updated node representations
        output = self.bellmanford(data, h_index[:, 0], r_index[:, 0])  # (num_nodes, batch_size, feature_dim）
        feature = output["node_feature"]
        index = t_index.unsqueeze(-1).expand(-1, -1, feature.shape[-1])
        # extract representations of tail entities from the updated node states
        feature = feature.gather(1, index)  # (batch_size, num_negative + 1, feature_dim)

        # probability logit for each tail node in the batch
        # (batch_size, num_negative + 1, dim) -> (batch_size, num_negative + 1)
        score = self.mlp(feature).squeeze(-1)
        return score.view(shape)

    
class NBFNet(BaseNBFNet):

    def __init__(self, input_dim, hidden_dims, num_relation, message_func="distmult", aggregate_func="sum",
                 short_cut=False, layer_norm=False, activation="relu", concat_hidden=False, num_mlp_layer=2,
                 dependent=True, remove_one_hop=False, num_beam=10, path_topk=10, **kwargs):
        super().__init__(input_dim, hidden_dims, num_relation, message_func=message_func, aggregate_func=aggregate_func,
                 short_cut=short_cut, layer_norm=layer_norm, activation=activation, concat_hidden=concat_hidden, num_mlp_layer=num_mlp_layer,
                 dependent=dependent, remove_one_hop=remove_one_hop, num_beam=num_beam, path_topk=path_topk, **kwargs)

        self.layers = nn.ModuleList()
        self.copy_weights = kwargs.get('copy_weights', False)
        self.freeze_relation_weights = kwargs.get('freeze_relation_weights', False)
        self.orthogonal = kwargs.get('orthogonal', False)

        hidden_dims = [input_dim]*(hidden_dims) if isinstance(hidden_dims, int) else hidden_dims

        for i in range(len(self.dims) - 1):
            if self.copy_weights:
                print('COPYING WEIGHTS...')
                if i == 0:
                    l0 = layers.GeneralizedRelationalConv(self.dims[i], self.dims[i + 1], num_relation,
                                                                self.dims[0], message_func, aggregate_func, layer_norm,
                                                                activation, dependent, orthogonal=self.orthogonal)
                layer = l0
            else:
                layer = layers.GeneralizedRelationalConv(self.dims[i], self.dims[i + 1], num_relation,
                                                                self.dims[0], message_func, aggregate_func, layer_norm,
                                                                activation, dependent, orthogonal=self.orthogonal)
            if self.freeze_relation_weights:
                for param in layer.parameters():
                    param.requires_grad = False    
            self.layers.append(layer)

        feature_dim = (sum(hidden_dims) if concat_hidden else hidden_dims[-1]) + input_dim

        # additional relation embedding which serves as an initial 'query' for the NBFNet forward pass
        # each layer has its own learnable relations matrix, so we send the total number of relations, too
        self.query = nn.Embedding(num_relation, input_dim)
        self.mlp = nn.Sequential()
        mlp = []
        for i in range(num_mlp_layer - 1):
            mlp.append(nn.Linear(feature_dim, feature_dim))
            mlp.append(nn.ReLU())
        mlp.append(nn.Linear(feature_dim, 1))
        self.mlp = nn.Sequential(*mlp)
    

    def forward(self, data, batch):
        h_index, t_index, r_index = batch.unbind(-1)
        if self.training:
            # Edge dropout in the training mode
            # here we want to remove immediate edges (head, relation, tail) from the edge_index and edge_types
            # to make NBFNet iteration learn non-trivial paths
            data = self.remove_easy_edges(data, h_index, t_index, r_index)

        shape = h_index.shape
        # turn all triples in a batch into a tail prediction mode
        h_index, t_index, r_index = self.negative_sample_to_tail(h_index, t_index, r_index, num_direct_rel=data.num_relations // 2)
        assert (h_index[:, [0]] == h_index).all()
        assert (r_index[:, [0]] == r_index).all()

        # message passing and updated node representations
        output = self.bellmanford(data, h_index[:, 0], r_index[:, 0])  # (num_nodes, batch_size, feature_dim）
        feature = output["node_feature"]
        index = t_index.unsqueeze(-1).expand(-1, -1, feature.shape[-1])
        # extract representations of tail entities from the updated node states
        feature = feature.gather(1, index)  # (batch_size, num_negative + 1, feature_dim)

        # probability logit for each tail node in the batch
        # (batch_size, num_negative + 1, dim) -> (batch_size, num_negative + 1)
        score = self.mlp(feature).squeeze(-1)
        return score.view(shape)


class NBFNetInv(NBFNet):

    def __init__(self, input_dim, hidden_dims, num_relation, base_model=None, message_func="distmult", aggregate_func="sum",
                 short_cut=False, layer_norm=False, activation="relu", concat_hidden=False, num_mlp_layer=2,
                 dependent=True, remove_one_hop=False, num_beam=10, path_topk=10, normalization='sym', 
                  atol=1e-6, threshold_ptile=0, copy_weights=True, **kwargs):
        super().__init__(input_dim, hidden_dims, num_relation, message_func=message_func, aggregate_func=aggregate_func,
                 short_cut=short_cut, layer_norm=layer_norm, activation=activation, concat_hidden=concat_hidden, num_mlp_layer=num_mlp_layer,
                 dependent=dependent, remove_one_hop=remove_one_hop, num_beam=num_beam, path_topk=path_topk, **kwargs)        

        feature_dim = (sum(hidden_dims) if concat_hidden else hidden_dims[-1]) + input_dim


        self.normalization = normalization
        self.atol = atol
        self.threshold_ptile = threshold_ptile
        self.copy_weights = copy_weights
        self.freeze_relation_weights = False
        self.Linv = None
        # additional relation embedding which serves as an initial 'query' for the NBFNet forward pass
        # each layer has its own learnable relations matrix, so we send the total number of relations, too
        
        if base_model is not None:
            self.mlp = base_model.mlp
            self.query = base_model.query
        else:
            if kwargs.get('freeze_relation_weights', False):
                self.query.requires_grad_(False)
                self.freeze_relation_weights = True
            self.mlp = nn.Sequential()
            mlp = []
            for i in range(num_mlp_layer - 1):
                mlp.append(nn.Linear(feature_dim, feature_dim))
                mlp.append(nn.ReLU())
            mlp.append(nn.Linear(feature_dim, 1))
            self.mlp = nn.Sequential(*mlp)

    def translate_to_graph_rep_inv(self, data):
        nbf_layers = self.layers
        if self.copy_weights:
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

            L_edge_index, L_edge_weight = get_laplacian(data.edge_index, edge_weight=rels_reshaped, 
                                                        normalization=self.normalization)
            L = to_dense_adj(L_edge_index, edge_attr=L_edge_weight, max_num_nodes=data.num_nodes)
            L = L[0] # take first (and only) in batch
            # Linv = L
            # for i in range(2,100):
            #     Linv += torch.linalg.matrix_power(I - L, i)[0]
            # print('computing inverse...')

            res = []
            for d in tqdm(range(rels_reshaped.shape[1])):
                Linv = torch.linalg.pinv(L[:,:,d], atol=self.atol)
                res.append(Linv.unsqueeze(-1).to('cpu')) # store on cpu to save memory

            res = torch.concatenate(res, axis=-1)
            res = res.to(rels.device)
            # I = torch.eye(data.num_nodes).to(rels.device)
            # I = I.unsqueeze(-1)
            # res = res - I

            if self.threshold_pctile > 0:
                # Reshape the tensor to collapse the first two dimensions
                reshaped_tensor = res.view(-1, res.size(-1))

                # Calculate the 95th percentile along the new dimension (0)
                percentiles = torch.quantile(reshaped_tensor, self.threshold_pctile, dim=0, keepdim=True)

                # Reshape percentiles to enable broadcasting (1, 1, number of channels)
                percentiles = percentiles.view(1, 1, res.size(-1))

                # Threshold the tensor: values below the 95th percentile are set to zero
                res = torch.where(res >= percentiles, res, torch.tensor(0.0))
                # percentiles = torch.quantile(res.to('cpu'), threshold_pctile, dim=None)
                # res = torch.where(res >= percentiles.to(res.device), res, torch.tensor(0.0))

            return res

    
    def init_lap(self, data):
        with torch.no_grad():
            self.Linv = self.translate_to_graph_rep_inv(self, data)
        if not self.freeze_relation_weights:
            self.Linv.requires_grad = True

    def forward(self, data, batch):
        h_index, t_index, r_index = batch.unbind(-1)
        if self.training:
            # Edge dropout in the training mode
            # here we want to remove immediate edges (head, relation, tail) from the edge_index and edge_types
            # to make NBFNet iteration learn non-trivial paths
            data = self.remove_easy_edges(data, h_index, t_index, r_index)

        shape = h_index.shape
        # turn all triples in a batch into a tail prediction mode
        h_index, t_index, r_index = self.negative_sample_to_tail(h_index, t_index, r_index, num_direct_rel=data.num_relations // 2)
        assert (h_index[:, [0]] == h_index).all()
        assert (r_index[:, [0]] == r_index).all()

        # message passing and updated node representations
        feature = torch.index_select(self.Linv, 0, h_index[:,0])
        query = torch.index_select(self.query.weight, 0, r_index[:,0])
        query = query.unsqueeze(1).expand((-1, feature.shape[1], -1))
        
        feature = torch.cat([feature, query], dim=-1)
        index = t_index.unsqueeze(-1).expand(-1, -1, feature.shape[-1])
        # extract representations of tail entities from the updated node states
        feature = feature.gather(1, index)  # (batch_size, num_negative + 1, feature_dim)

        # probability logit for each tail node in the batch
        # (batch_size, num_negative + 1, dim) -> (batch_size, num_negative + 1)
        score = self.mlp(feature).squeeze(-1)
        return score.view(shape)
    

class NBFNetEig(NBFNet):

    def __init__(self, input_dim, hidden_dims, num_relation, message_func="distmult", aggregate_func="sum",
                 short_cut=False, layer_norm=False, activation="relu", concat_hidden=False, num_mlp_layer=2,
                 dependent=True, remove_one_hop=False, num_beam=10, path_topk=10, normalization='sym', 
                  k=16, atol=1e-6, niter=8, copy_weights=True, inv_device='cpu', **kwargs):
        super().__init__(input_dim, hidden_dims, num_relation, message_func=message_func, aggregate_func=aggregate_func,
                 short_cut=short_cut, layer_norm=layer_norm, activation=activation, concat_hidden=concat_hidden, num_mlp_layer=num_mlp_layer,
                 dependent=dependent, remove_one_hop=remove_one_hop, num_beam=num_beam, path_topk=path_topk, **kwargs)        
        

        feature_dim = (sum(hidden_dims) if concat_hidden else hidden_dims[-1]) + input_dim

        self.normalization = normalization
        self.k = k
        self.atol = atol
        self.niter = niter
        self.copy_weights = copy_weights
        self.freeze_relation_weights = False
        self.Leig = None
        self.inv_device = torch.device(inv_device)
        # additional relation embedding which serves as an initial 'query' for the NBFNet forward pass
        # each layer has its own learnable relations matrix, so we send the total number of relations, too
        
        self.query = nn.Embedding(num_relation, input_dim)
        if kwargs.get('freeze_query_weights', False):
            self.freeze_query_weights = True
            self.query.requires_grad_(False)
            
        self.mlp = nn.Sequential()
        mlp = []
        for i in range(num_mlp_layer - 1):
            mlp.append(nn.Linear(feature_dim, feature_dim))
            mlp.append(nn.ReLU())
        mlp.append(nn.Linear(feature_dim, 1))
        self.mlp = nn.Sequential(*mlp)

    def translate_to_graph_rep_eig(self, data):
        nbf_layers = self.layers
        if self.copy_weights:
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
            L_edge_index, L_edge_weight = get_laplacian(data.edge_index, edge_weight=rels_reshaped, 
                                                        normalization=self.normalization)

            res = []
            for d in tqdm(range(L_edge_weight.shape[1])):
                # L = to_dense_adj(L_edge_index, edge_attr=L_edge_weight[:,d], max_num_nodes=data.num_nodes)
                # L = L[0] # take first (and only) in batch
                # eig_vals, eig_vecs = torch.lobpcg(L, k=self.k, 
                #                                 largest=False, method='ortho',
                #                                 tol=self.atol, niter=self.niter)
                L = torch.sparse_coo_tensor(L_edge_index, L_edge_weight[:,d], 
                                            (data.num_nodes, data.num_nodes))
                eig_vals, eig_vecs = torch.lobpcg(L, k=self.k, 
                                                largest=False, method='ortho',
                                                tol=self.atol, niter=self.niter)

                eig_vecs = torch.real(eig_vecs)
                eig_vals = torch.real(eig_vals)
                
                eig_vals[eig_vals > self.atol] = 1/eig_vals[eig_vals > self.atol]
                eig_vals[eig_vals < self.atol] = 0
                pe = eig_vecs * eig_vals @ eig_vecs.T

                res.append(pe.unsqueeze(-1).to(self.inv_device))
        res = torch.concatenate(res, axis=-1)

        return res
    
    def init_lap(self, data):
        with torch.no_grad():
            self.Leig = self.translate_to_graph_rep_eig(data)
        if not self.freeze_relation_weights:
            self.Leig.requires_grad = True

    def forward(self, data, batch):
        h_index, t_index, r_index = batch.unbind(-1)
        if self.training:
            # Edge dropout in the training mode
            # here we want to remove immediate edges (head, relation, tail) from the edge_index and edge_types
            # to make NBFNet iteration learn non-trivial paths
            data = self.remove_easy_edges(data, h_index, t_index, r_index)

        shape = h_index.shape
        # turn all triples in a batch into a tail prediction mode
        h_index, t_index, r_index = self.negative_sample_to_tail(h_index, t_index, r_index, num_direct_rel=data.num_relations // 2)
        assert (h_index[:, [0]] == h_index).all()
        assert (r_index[:, [0]] == r_index).all()

        # message passing and updated node representations
        if self.inv_device != h_index.device:
            feature = torch.index_select(self.Leig, 0, h_index[:,0].to(self.inv_device)).to(r_index.device)
        else:
            feature = torch.index_select(self.Leig, 0, h_index[:,0])
        query = torch.index_select(self.query.weight, 0, r_index[:,0])
        query = query.unsqueeze(1).expand((-1, feature.shape[1], -1))
        
        feature = torch.cat([feature, query], dim=-1)
        index = t_index.unsqueeze(-1).expand(-1, -1, feature.shape[-1])
        # extract representations of tail entities from the updated node states
        feature = feature.gather(1, index)  # (batch_size, num_negative + 1, feature_dim)

        score = self.mlp(feature).squeeze(-1)
        return score.view(shape)


class NBFNetDepEig(nn.Module):

    def __init__(self, input_dim, num_relation, message_func="distmult", aggregate_func="sum",
                 short_cut=False, layer_norm=False, activation="relu", concat_hidden=False, num_mlp_layer=2,
                 dependent=True, remove_one_hop=False, num_beam=10, normalization='sym', 
                  k=16, atol=1e-6, niter=8, copy_weights=True, **kwargs):
        super().__init__()
        
        feature_dim = input_dim
        self.feature_dim = feature_dim
        self.normalization = normalization
        self.k = k
        self.atol = atol
        self.niter = niter
        self.copy_weights = copy_weights
        self.freeze_relation_weights = False
        self.Leig = None
        self.dependent = kwargs.get('dependent', False)
        # additional relation embedding which serves as an initial 'query' for the NBFNet forward pass
        # each layer has its own learnable relations matrix, so we send the total number of relations, too
        
        self.query = nn.Embedding(num_relation, input_dim)
        if kwargs.get('freeze_relation_weights', False):
            self.query.requires_grad_(False)
            self.freeze_relation_weights = True
        self.mlp = nn.Sequential()
        mlp = []
        for i in range(num_mlp_layer - 1):
            mlp.append(nn.Linear(feature_dim*2, feature_dim*2))
            mlp.append(nn.ReLU())
        mlp.append(nn.Linear(feature_dim*2, 1))
        self.mlp = nn.Sequential(*mlp)

    def remove_easy_edges(self, data, h_index, t_index, r_index=None):
        # we remove training edges (we need to predict them at training time) from the edge index
        # think of it as a dynamic edge dropout
        h_index_ext = torch.cat([h_index, t_index], dim=-1)
        t_index_ext = torch.cat([t_index, h_index], dim=-1)
        r_index_ext = torch.cat([r_index, r_index + data.num_relations // 2], dim=-1)
        
        # we remove existing immediate edges between heads and tails in the batch with the given relation
        edge_index = torch.cat([data.edge_index, data.edge_type.unsqueeze(0)])
        # note that here we add relation types r_index_ext to the matching query
        easy_edge = torch.stack([h_index_ext, t_index_ext, r_index_ext]).flatten(1)
        index = tasks.edge_match(edge_index, easy_edge)[0]
        mask = ~index_to_mask(index, data.num_edges)

        data = copy.copy(data)
        data.edge_index = data.edge_index[:, mask]
        data.edge_type = data.edge_type[mask]
        return data

    def negative_sample_to_tail(self, h_index, t_index, r_index, num_direct_rel):
        # convert p(h | t, r) to p(t' | h', r')
        # h' = t, r' = r^{-1}, t' = h
        is_t_neg = (h_index == h_index[:, [0]]).all(dim=-1, keepdim=True)
        new_h_index = torch.where(is_t_neg, h_index, t_index)
        new_t_index = torch.where(is_t_neg, t_index, h_index)
        new_r_index = torch.where(is_t_neg, r_index, r_index + num_direct_rel)
        return new_h_index, new_t_index, new_r_index

    def translate_to_graph_rep_eig(self, data):
        
        rels = torch.ones((data.num_relations, self.feature_dim), device=data.edge_index.device)
        ret = []
        for reltype in tqdm(range(data.num_relations)):
            
            # now we're assuming the relation representation is a vector, implying 
            # the associated sheaf Laplacian is the direct sum of the Laplacian in 
            # each dimension
            relmsk = (data.edge_type == reltype)
            rels_reshaped = torch.index_select(rels, 0, data.edge_type[relmsk])
            
            L_edge_index, L_edge_weight = get_laplacian(data.edge_index[:,relmsk], edge_weight=rels_reshaped, 
                                                        normalization=self.normalization)

            res = []
            
            L = to_dense_adj(L_edge_index, edge_attr=L_edge_weight, max_num_nodes=data.num_nodes)
            L = L[0].permute(-1, 0, 1) # take first (and only) in batch

            for d in range(L.shape[0]):

                eig_vals, eig_vecs = torch.lobpcg(L[d], k=self.k, 
                                                largest=False, method='ortho',
                                                tol=self.atol, niter=self.niter)

                eig_vecs = torch.real(eig_vecs)
                eig_vals = torch.real(eig_vals)
                
                eig_vals[eig_vals > self.atol] = 1/eig_vals[eig_vals > self.atol]
                eig_vals[eig_vals < self.atol] = 0
                pe = eig_vecs * eig_vals @ eig_vecs.T

                res.append(pe.unsqueeze(-1))
            res = torch.concatenate(res, axis=-1)
            ret.append(res.unsqueeze(0))
        ret = torch.concatenate(ret, axis=0).to(rels.device)
        return ret
    
    def init_lap(self, data):
        with torch.no_grad():
            self.Leig = self.translate_to_graph_rep_eig(data)
        if not self.freeze_relation_weights:
            self.Leig.requires_grad = True

    def forward(self, data, batch):
        h_index, t_index, r_index = batch.unbind(-1)
        if self.training:
            # Edge dropout in the training mode
            # here we want to remove immediate edges (head, relation, tail) from the edge_index and edge_types
            # to make NBFNet iteration learn non-trivial paths
            data = self.remove_easy_edges(data, h_index, t_index, r_index)

        shape = h_index.shape
        # turn all triples in a batch into a tail prediction mode
        h_index, t_index, r_index = self.negative_sample_to_tail(h_index, t_index, r_index, num_direct_rel=data.num_relations // 2)
        assert (h_index[:, [0]] == h_index).all()
        assert (r_index[:, [0]] == r_index).all()

        # message passing and updated node representations
        feature = self.Leig[r_index[:,0], h_index[:,0]]
        query = torch.index_select(self.query.weight, 0, r_index[:,0])
        query = query.unsqueeze(1).expand((-1, feature.shape[1], -1))
        
        feature = torch.cat([feature, query], dim=-1)
        index = t_index.unsqueeze(-1).expand(-1, -1, feature.shape[-1])
        # extract representations of tail entities from the updated node states
        feature = feature.gather(1, index)  # (batch_size, num_negative + 1, feature_dim)

        score = self.mlp(feature).squeeze(-1)
        return score.view(shape)
    

class NTNEig(NBFNet):

    def __init__(self, input_dim, hidden_dims, num_relation, message_func="distmult", aggregate_func="sum",
                 short_cut=False, layer_norm=False, activation="relu", concat_hidden=False, num_mlp_layer=2,
                 dependent=True, remove_one_hop=False, num_beam=10, path_topk=10, normalization='sym', 
                  k=16, atol=1e-6, niter=8, copy_weights=True, **kwargs):
        super().__init__(input_dim, hidden_dims, num_relation, message_func=message_func, aggregate_func=aggregate_func,
                 short_cut=short_cut, layer_norm=layer_norm, activation=activation, concat_hidden=concat_hidden, num_mlp_layer=num_mlp_layer,
                 dependent=dependent, remove_one_hop=remove_one_hop, num_beam=num_beam, path_topk=path_topk, **kwargs)        
        

        feature_dim = (sum(hidden_dims) if concat_hidden else hidden_dims[-1]) + input_dim

        self.normalization = normalization
        self.k = k
        self.atol = atol
        self.niter = niter
        self.copy_weights = copy_weights
        self.freeze_relation_weights = False
        self.Leig = None
        # additional relation embedding which serves as an initial 'query' for the NBFNet forward pass
        # each layer has its own learnable relations matrix, so we send the total number of relations, too
        
        self.query = nn.Embedding(num_relation, input_dim)
        if kwargs.get('freeze_relation_weights', False):
            self.query.requires_grad_(False)
            self.freeze_relation_weights = True
        self.mlp = nn.Sequential()
        mlp = []
        for i in range(num_mlp_layer - 1):
            mlp.append(nn.Linear(feature_dim, feature_dim))
            mlp.append(nn.ReLU())
        mlp.append(nn.Linear(feature_dim, 1))
        self.mlp = nn.Sequential(*mlp)

    def translate_to_graph_rep_eig(self, data):
        nbf_layers = self.layers
        if self.copy_weights:
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

            L_edge_index, L_edge_weight = get_laplacian(data.edge_index, edge_weight=rels_reshaped, 
                                                        normalization=self.normalization)

            res = []
            
            L = to_dense_adj(L_edge_index, edge_attr=L_edge_weight, max_num_nodes=data.num_nodes)
            L = L[0].permute(-1, 0, 1) # take first (and only) in batch

            for d in tqdm(range(L.shape[0])):

                eig_vals, eig_vecs = torch.lobpcg(L[d], k=self.k, 
                                                largest=False, method='ortho',
                                                tol=self.atol, niter=self.niter)

                eig_vecs = torch.real(eig_vecs)
                eig_vals = torch.real(eig_vals)
                
                eig_vals[eig_vals > self.atol] = 1/eig_vals[eig_vals > self.atol]
                eig_vals[eig_vals < self.atol] = 0
                pe = eig_vecs * eig_vals @ eig_vecs.T

                res.append(pe.unsqueeze(-1))
        res = torch.concatenate(res, axis=-1).to(rels.device)

        return res
    
    def init_lap(self, data):
        with torch.no_grad():
            self.Leig = self.translate_to_graph_rep_eig(data)
        if not self.freeze_relation_weights:
            self.Leig.requires_grad = True

    def forward(self, data, batch):
        h_index, t_index, r_index = batch.unbind(-1)
        if self.training:
            # Edge dropout in the training mode
            # here we want to remove immediate edges (head, relation, tail) from the edge_index and edge_types
            # to make NBFNet iteration learn non-trivial paths
            data = self.remove_easy_edges(data, h_index, t_index, r_index)

        shape = h_index.shape
        # turn all triples in a batch into a tail prediction mode
        h_index, t_index, r_index = self.negative_sample_to_tail(h_index, t_index, r_index, num_direct_rel=data.num_relations // 2)
        assert (h_index[:, [0]] == h_index).all()
        assert (r_index[:, [0]] == r_index).all()

        # message passing and updated node representations
        feature = torch.index_select(self.Leig, 0, h_index[:,0])
        query = torch.index_select(self.query.weight, 0, r_index[:,0])
        query = query.unsqueeze(1).expand((-1, feature.shape[1], -1))
        
        feature = torch.cat([feature, query], dim=-1)
        index = t_index.unsqueeze(-1).expand(-1, -1, feature.shape[-1])
        # extract representations of tail entities from the updated node states
        feature = feature.gather(1, index)  # (batch_size, num_negative + 1, feature_dim)

        score = self.mlp(feature).squeeze(-1)
        return score.view(shape)