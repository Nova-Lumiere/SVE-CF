import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.sparse import coo_matrix, eye
from torch_geometric.utils import degree

from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType

from recbole_gnn.model.abstract_recommender import SocialRecommender
from recbole_gnn.model.layers import LightGCNConv, BipartiteGCNConv

class GatingLayer(nn.Module):
    def __init__(self, dim):
        super(GatingLayer, self).__init__()
        self.dim = dim
        self.linear = nn.Linear(self.dim, self.dim)
        self.activation = nn.Sigmoid()

    def forward(self, emb):
        embedding = self.linear(emb)
        embedding = self.activation(embedding)
        embedding = torch.mul(emb, embedding)
        return embedding

class svecf(SocialRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(svecf, self).__init__(config, dataset)

        # load dataset info
        self.edge_index, self.edge_weight = dataset.get_norm_adj_mat()
        self.edge_index, self.edge_weight = self.edge_index.to(self.device), self.edge_weight.to(self.device)
        M_s, M_j = self.get_motif_adj_matrix(dataset)

        # transform matrix to edge index and edge weight for convolution
        self.M_s_edge_index, self.M_s_edge_weight = self.get_edge_index_weight(M_s)
        self.M_j_edge_index, self.M_j_edge_weight = self.get_edge_index_weight(M_j)

        self._user = dataset.inter_feat[dataset.uid_field]
        self._item = dataset.inter_feat[dataset.iid_field]

        self._src_user = dataset.net_feat[dataset.net_src_field]
        self._tgt_user = dataset.net_feat[dataset.net_tgt_field]

        # load parameters info
        self.latent_dim = config["latent_dim"]
        self.n_layers = int(config["n_layers"])
        self.drop_ratio = config["drop_ratio"]
        self.instance_cnt = config["instance_cnt"]
        self.reg_weight = config["reg_weight"]
        self.ssl_weight = config["ssl_weight"]
        self.ssl_tau = config["ssl_tau"]

        # define layers and loss
        self.user_embedding = torch.nn.Embedding(self.n_users, self.latent_dim)
        self.item_embedding = torch.nn.Embedding(self.n_items, self.latent_dim)
        self.gcn_conv = LightGCNConv(dim=self.latent_dim)
        self.bipartite_gcn_conv = BipartiteGCNConv(dim=self.latent_dim)
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # define gating layers
        self.gating_c1 = GatingLayer(self.latent_dim)
        self.gating_c2 = GatingLayer(self.latent_dim)
        
        # storage variables for full sort evaluation acceleration
        self.user_all_embeddings = None
        self.user_embeddings_follow = None
        self.user_embeddings_joint = None
        self.restore_user_e = None
        self.restore_item_e = None

        # parameters initialization
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ['restore_user_e', 'restore_item_e']

    def get_norm_edge_weight(self, edge_index, node_num):
        r"""Get normalized edge weight using the laplace matrix.
        """
        deg = degree(edge_index[0], node_num)
        norm_deg = 1. / torch.sqrt(torch.where(deg == 0, torch.ones([1]), deg))
        edge_weight = norm_deg[edge_index[0]] * norm_deg[edge_index[1]]
        return edge_weight


    def subgraph_construction(self):
        r"""Perturb the joint graph to construct subgraph for integrated self-supervision signals.
        """
        def rand_sample(high, size=None, replace=True):
            return np.random.choice(np.arange(high), size=size, replace=replace)
        # perturb the raw graph with edge dropout
        keep = rand_sample(len(self._user), size=int(len(self._user) * (1 - self.drop_ratio)), replace=False)
        row = self._user[keep]
        col = self._item[keep] + self.n_users

        # perturb the social graph with edge dropout
        net_keep = rand_sample(len(self._src_user), size=int(len(self._src_user) * (1 - self.drop_ratio)), replace=False)
        net_row = self._src_user[net_keep]
        net_col = self._tgt_user[net_keep]

        # concatenation and normalization
        edge_index1 = torch.stack([row, col])
        edge_index2 = torch.stack([col, row])
        edge_index3 = torch.stack([net_row, net_col])
        edge_index = torch.cat([edge_index1, edge_index2, edge_index3], dim=1)
        edge_weight = self.get_norm_edge_weight(edge_index, self.n_users + self.n_items)

        self.sub_graph = edge_index.to(self.device), edge_weight.to(self.device)


    def get_motif_adj_matrix(self, dataset):
        S = dataset.net_matrix()
        Y = dataset.inter_matrix()
        B = S.multiply(S.T)
        U = S - B
        C1 = (U.dot(U)).multiply(U.T)
        A1 = C1 + C1.T
        C2 = (B.dot(U)).multiply(U.T) + (U.dot(B)).multiply(U.T) + (U.dot(U)).multiply(B)
        A2 = C2 + C2.T
        C3 = (B.dot(B)).multiply(U) + (B.dot(U)).multiply(B) + (U.dot(B)).multiply(B)
        A3 = C3 + C3.T
        A4 = (B.dot(B)).multiply(B)
        C5 = (U.dot(U)).multiply(U) + (U.dot(U.T)).multiply(U) + (U.T.dot(U)).multiply(U)
        A5 = C5 + C5.T
        A6 = (U.dot(B)).multiply(U) + (B.dot(U.T)).multiply(U.T) + (U.T.dot(U)).multiply(B)
        A7 = (U.T.dot(B)).multiply(U.T) + (B.dot(U)).multiply(U) + (U.dot(U.T)).multiply(B)
        A8 = (Y.dot(Y.T)).multiply(B)
        A9 = (Y.dot(Y.T)).multiply(U)
        A9 = A9 + A9.T
        A10  = Y.dot(Y.T) - A8 - A9
        # addition and row-normalization
        M_s = sum([A1, A2, A3, A4, A5, A6, A7])
        # add epsilon to avoid divide by zero Warning
        M_s = M_s.multiply(1.0 / (M_s.sum(axis=1) + 1e-7).reshape(-1, 1))
        M_j = sum([A8, A9, A10])
        M_j = M_j.multiply(1.0 / (M_j.sum(axis=1) + 1e-7).reshape(-1, 1))

        return M_s, M_j

    def forward(self, graph=None):
        # get ego embeddings
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight

        # self-gating
        user_embeddings_follow = self.gating_c1(user_embeddings)
        user_embeddings_joint = self.gating_c2(user_embeddings)
        user_embeddings_follow_list = [user_embeddings_follow]
        user_embeddings_joint_list = [user_embeddings_joint]

        all_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight])
        embeddings_list = [all_embeddings]

        if graph is None:  # for the multi-view graph
            edge_index, edge_weight = self.edge_index, self.edge_weight

            for _ in range(self.n_layers):
                all_embeddings = self.gcn_conv(all_embeddings, edge_index, edge_weight)
                norm_embeddings = F.normalize(all_embeddings, p=2, dim=1)
                embeddings_list.append(norm_embeddings)

                # follow view
                user_embeddings_follow = self.bipartite_gcn_conv((user_embeddings_follow, user_embeddings_follow),
                                                                 self.M_s_edge_index.flip([0]), self.M_s_edge_weight,
                                                                 size=(self.n_users, self.n_users))
                norm_embeddings = F.normalize(user_embeddings_follow, p=2, dim=1)
                user_embeddings_follow_list += [norm_embeddings]

                # joint view
                user_embeddings_joint = self.bipartite_gcn_conv((user_embeddings_joint, user_embeddings_joint),
                                                                self.M_j_edge_index.flip([0]), self.M_j_edge_weight,
                                                                size=(self.n_users, self.n_users))
                norm_embeddings = F.normalize(user_embeddings_joint, p=2, dim=1)
                user_embeddings_joint_list += [norm_embeddings]

            # averaging the view-specific embeddings
            user_embeddings_follow = torch.stack(user_embeddings_follow_list, dim=0).sum(dim=0)
            user_embeddings_joint = torch.stack(user_embeddings_joint_list, dim=0).sum(dim=0)

            all_embeddings = torch.stack(embeddings_list, dim=1)
            all_embeddings = torch.sum(all_embeddings, dim=1)
            user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)

        else:  # for the augmented graph
            edge_index, edge_weight = graph

            for _ in range(self.n_layers):
                all_embeddings = self.gcn_conv(all_embeddings, edge_index, edge_weight)
                norm_embeddings = F.normalize(all_embeddings, p=2, dim=1)
                embeddings_list.append(norm_embeddings)

            all_embeddings = torch.stack(embeddings_list, dim=1)
            all_embeddings = torch.sum(all_embeddings, dim=1)
            user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)

        if graph is None:
            return user_all_embeddings, item_all_embeddings, user_embeddings_follow, user_embeddings_joint
        else:
            return user_all_embeddings

    def label_prediction(self, emb, aug_emb):
        prob = torch.matmul(emb, aug_emb.transpose(0, 1))
        prob = F.softmax(prob, dim=1)
        return prob

    def sampling(self, logits):
        return torch.topk(logits, k=self.instance_cnt)[1]

    def generate_pesudo_labels(self, prob1, prob2):
        positive = (prob1 + prob2) / 2
        pos_examples = self.sampling(positive)
        return pos_examples

    def calculate_ssl_loss(self, aug_emb, positive, emb):
        pos_emb = aug_emb[positive]
        pos_score = torch.sum(emb.unsqueeze(dim=1).repeat(1, self.instance_cnt, 1) * pos_emb, dim=2)
        ttl_score = torch.matmul(emb, aug_emb.transpose(0, 1))
        pos_score = torch.sum(torch.exp(pos_score / self.ssl_tau), dim=1)
        ttl_score = torch.sum(torch.exp(ttl_score / self.ssl_tau), dim=1)
        ssl_loss = - torch.sum(torch.log(pos_score / ttl_score))
        return ssl_loss

    def calculate_rec_loss(self, interaction):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        self.user_all_embeddings, item_all_embeddings,_, _ = self.forward()
        u_embeddings = self.user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]
        neg_embeddings = item_all_embeddings[neg_item]

        # calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        # calculate regularization Loss
        u_ego_embeddings = self.user_embedding(user)
        pos_ego_embeddings = self.item_embedding(pos_item)
        neg_ego_embeddings = self.item_embedding(neg_item)

        reg_loss = self.reg_loss(u_ego_embeddings, pos_ego_embeddings, neg_ego_embeddings)
        loss = mf_loss + self.reg_weight * reg_loss

        return loss

    def calculate_loss(self, interaction):
        # preference view
        rec_loss = self.calculate_rec_loss(interaction)

        # unlabeled sample view
        aug_user_embeddings = self.forward(graph=self.sub_graph)

        # multi-view calculation
        self.user_all_embeddings, item_all_embeddings, self.user_embeddings_follow, self.user_embeddings_joint = self.forward()

        user = interaction[self.USER_ID]
        aug_u_embeddings = aug_user_embeddings[user]
        follow_u_embeddings = self.user_embeddings_follow[user]
        joint_u_embeddings = self.user_embeddings_joint[user]
        rec_u_embeddings = self.user_all_embeddings[user]

        aug_u_embeddings = F.normalize(aug_u_embeddings, p=2, dim=1)
        follow_u_embeddings = F.normalize(follow_u_embeddings, p=2, dim=1)
        joint_u_embeddings = F.normalize(joint_u_embeddings, p=2, dim=1)
        rec_u_embeddings = F.normalize(rec_u_embeddings, p=2, dim=1)

        # self-supervision prediction
        follow_prediction = self.label_prediction(follow_u_embeddings, aug_u_embeddings)
        joint_prediction = self.label_prediction(joint_u_embeddings, aug_u_embeddings)
        rec_prediction = self.label_prediction(rec_u_embeddings, aug_u_embeddings)

        # find informative positive examples for each encoder
        follow_pos = self.generate_pesudo_labels(follow_prediction, rec_prediction)
        joint_pos = self.generate_pesudo_labels(joint_prediction, rec_prediction)
        rec_pos = self.generate_pesudo_labels(follow_prediction, joint_prediction)

        # neighbor-discrimination based contrastive learning  tri-view joint learning
        ssl_loss = self.calculate_ssl_loss(aug_u_embeddings, follow_pos, follow_u_embeddings)
        ssl_loss += self.calculate_ssl_loss(aug_u_embeddings, joint_pos, joint_u_embeddings)
        ssl_loss += self.calculate_ssl_loss(aug_u_embeddings, rec_pos, rec_u_embeddings)

        # L = L_r + β * L_{ssl}
        loss = rec_loss + self.ssl_weight * ssl_loss

        return loss

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]

        user_all_embeddings, item_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        # get user embedding from storage variable
        u_embeddings = self.restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores.view(-1)