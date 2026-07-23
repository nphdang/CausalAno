import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform

class base_continuous_generator(nn.Module):
    def __init__(self, parent_dim, z_dim, feature_dim):
        super(base_continuous_generator, self).__init__()
        self.parent_dim = parent_dim
        self.z_dim = z_dim

        self.feature_dim = feature_dim

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(self.parent_dim+self.z_dim, 64, normalize=False),
            *block(64, 128),
            # *block(128, 128),
            *block(128, 128),
            nn.Linear(128, self.feature_dim)
        )
        # std cố định tạm thời
        self.log_std = nn.Parameter(torch.zeros(1))  # std = exp(log_std)

    def forward(self, noise, parents):
        x = torch.cat([parents, noise], dim=-1) if parents is not None else noise
        x = self.model(x)

        if self.feature_dim == 1:
            x = torch.tanh(x)
        else:
            x_t = []
            x_t.append(torch.tanh(x[:, 0]).unsqueeze(dim=-1))
            x_t.append(nn.functional.gumbel_softmax(x[:, 1:], tau=0.2, hard=False, eps=1e-10, dim=-1))

            x = torch.cat(x_t, dim=1)
        return x

    @torch.no_grad()  # dùng khi chỉ muốn sinh không gradient (ví dụ buffer PC)
    def sample_only(self, noise, parents):
        """Sinh sample theo cùng phân phối như sample_with_logprob nhưng không trả log_prob (tiện D-step)."""
        a, _ = self.sample_with_logprob(noise, parents)
        return a

    def sample_with_logprob(self, noise, parents):
        """
        Sinh action + log_prob cho REINFORCE (continuous-only).
        - Continuous: TransformedDistribution(Normal, TanhTransform) -> action in [-1, 1], log_prob có hiệu chỉnh.
        """
        x_in = torch.cat([parents, noise], dim=-1) if parents is not None else noise
        raw = self.model(x_in)  # [B, feature_dim]
        std = torch.exp(self.log_std)  # broadcast được nếu self.log_std.shape == (1,)

        # Gaussian -> Tanh (áp dụng cho toàn bộ chiều continuous)
        base = Normal(loc=raw, scale=std)  # [B, feature_dim] với broadcast
        dist = TransformedDistribution(base, [TanhTransform(cache_size=1)])
        a = dist.rsample()  # [B, feature_dim] trong [-1, 1]
        logp = dist.log_prob(a).sum(dim=-1)  # [B] (sum theo feature_dim)
        return a, logp

class base_catogory_generator(nn.Module):
    def __init__(self, parent_dim, z_dim, feature_dim):
        super(base_catogory_generator, self).__init__()
        self.parent_dim = parent_dim
        self.z_dim = z_dim

        self.feature_dim = feature_dim

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(self.parent_dim+self.z_dim, 64, normalize=False),
            *block(64, 128),
            # *block(128, 128),
            *block(128, 128),
            nn.Linear(128, self.feature_dim)
        )

    def forward(self, noise, parents):
        x = torch.cat([parents, noise], dim=-1) if parents is not None else noise
        x = self.model(x)
        if self.feature_dim == 1:
            x = torch.relu(x)
        else:
            x = nn.functional.gumbel_softmax(x, tau=0.2, hard=False, eps=1e-10, dim=-1)

        return x

    @torch.no_grad()
    def sample_only(self, noise, parents):
        y, _ = self.sample_with_logprob(noise, parents)
        return y

    def sample_with_logprob(self, noise, parents):
        x_in = torch.cat([parents, noise], dim=-1) if parents is not None else noise
        logits = self.model(x_in)  # [B, K], K >= 2
        # manual Gumbel-Softmax sample (như CTGAN/Causal-TGAN)
        y = torch.nn.functional.gumbel_softmax(logits, tau=0.2, hard=False, dim=-1)  # [B, K]
        logp = (y.detach() * torch.nn.functional.log_softmax(logits, dim=-1)).sum(-1)  # [B]
        return y, logp

def get_continuous_generator(parent_dim, z_dim, feature_dim):
    return base_continuous_generator(parent_dim, z_dim, feature_dim)

def get_catogory_generator(parent_dim, z_dim, feature_dim):
    return base_catogory_generator(parent_dim, z_dim, feature_dim)

class CausalNode(object):
    """
    A node in causal graph.
    Fields in each node: parents node info; causal mechanism (nn.Module)
    """
    def __init__(self, device, z_dim, name, parents, feature_info):
        """
        :param parents: a list of names of parents nodes
        :param z_dim: dim_exogenous + dim_confounder
        """
        self.feature_dim = feature_info.dim_info[name]
        self.parents = parents
        self.parent_dim = sum([feature_info.dim_info[item] for item in parents]) if parents != [] else 0
        self.z_dim = z_dim
        self.device = device
        if feature_info.type_info[name] == 'continuous':
            self.causal_mechanism = get_continuous_generator(self.parent_dim, self.z_dim, self.feature_dim).to(self.device)
        else: # feature_info.type_info[name] = 'discrete'
            self.causal_mechanism = get_catogory_generator(self.parent_dim, self.z_dim, self.feature_dim).to(self.device)
        self.val = None

    def cal_val(self, noises, parents):
        """
        calculate the value of the nodes given its parents
        :param parents: list: parents values. This var is different from self.parents
        :return: the value of this node given its parents
        """
        self.val = self.causal_mechanism(noises, parents)

        return self.val

    def load_checkpoint(self, checkpoint):
        self.causal_mechanism.load_state_dict(checkpoint, strict=False)

    def fetch_checkpoint(self):
        return self.causal_mechanism.state_dict()

    def sample_with_logprob(self, noises, parents):
        """Trả về (val, logp) từ cơ chế của node này."""
        return self.causal_mechanism.sample_with_logprob(noises, parents)

class causal_generator(object):
    """
    Define the generator of CausalTGAN.
    Attributes: a causal graph that contains several CausalNode objects

    A causal graph (config.graph) is specified as follows:
            a list of pairs of (node, node_parents).
            Note that, the order of node names in graph must be consistent with the their order in the dataframe columns.
            Example: A->B<-C; D->E
            [ ['A',[]],
              ['B',['A','C']],
              ['C',[]],
              ['D',[]],
              ['E',['D']]
            ]

    """
    def __init__(self, device, config, feature_info):
        self.config = config
        self.device = device
        self.causal_graph = config.causal_graph
        self.keys = [self.causal_graph[i][0] for i in range(len(self.causal_graph))]
        self.feature_info = feature_info
        self.init_nodes()

    def init_nodes(self):
        self.nodes = {}
        for node_name, parent_list in self.causal_graph:
            z_dim = self.config.z_dim # exogenous dim
            self.nodes[node_name] = CausalNode(self.device, z_dim, node_name, parent_list, self.feature_info)
        # topology sorting
        self.name2idx = self.node_order()
        self.idx2name = dict((v, k) for k, v in self.name2idx.items())

    def sample(self, batch_size):
        """
        Sampling from causal graphs in autoregressive way
        :param batch_size: number of samples to generate
        :return: generated samples
        """
        fake_sample = torch.zeros((batch_size, sum(self.feature_info.dim_info.values()))).to(self.device)
        for idx in range(len(self.nodes)):
            exogenous_var = torch.Tensor(np.random.normal(size=(batch_size, self.config.z_dim))).to(self.device)
            current_node = self.nodes[self.idx2name[idx]]
            parents_name = current_node.parents
            parents_idx = self.feature_info.get_position_by_name(parents_name)  # get feature position (column index) in dataset
            parents_val = fake_sample[:, parents_idx] if parents_idx != [] else None
            val_position = self.feature_info.get_position_by_name(self.idx2name[idx])

            fake_sample[:, val_position] = current_node.cal_val(exogenous_var, parents_val)

        return fake_sample

    def sample_with_logprob(self, batch_size):
        """
        Sinh mẫu theo topo và cộng dồn log_prob toàn hàng: sum_logp_per_sample [B].
        Dùng cho bước REINFORCE (có gradient).
        """
        B = batch_size
        F = sum(self.feature_info.dim_info.values())
        fake_sample = torch.zeros((B, F), device=self.device)
        sum_logp = torch.zeros(B, device=self.device)

        for idx in range(len(self.nodes)):
            exo = torch.randn(B, self.config.z_dim, device=self.device)
            cur = self.nodes[self.idx2name[idx]]
            parents_name = cur.parents
            parents_idx = self.feature_info.get_position_by_name(parents_name)
            parents_val = fake_sample[:, parents_idx] if parents_idx != [] else None
            val_pos = self.feature_info.get_position_by_name(self.idx2name[idx])

            val, logp = cur.sample_with_logprob(exo, parents_val)  # [B, dim_i], [B]
            fake_sample[:, val_pos] = val
            sum_logp = sum_logp + logp

        return fake_sample, sum_logp  # [B, F], [B]

    def sample_anomalies(self, batch_size, corruption_prob=0.5):
        """
        Generate hard-to-detect causal anomalies by shuffling parents.
        """
        fake_sample = torch.zeros((batch_size, sum(self.feature_info.dim_info.values()))).to(self.device)
        labels = torch.zeros(batch_size).to(self.device)  # 0=Normal, 1=Anomaly

        # Determine which samples will be anomalies
        # (We corrupt a subset of the batch)
        is_anomaly = (torch.rand(batch_size) < corruption_prob).to(self.device)
        labels[is_anomaly] = 1.0

        for idx in range(len(self.nodes)):
            exogenous_var = torch.Tensor(np.random.normal(size=(batch_size, self.config.z_dim))).to(self.device)
            current_node = self.nodes[self.idx2name[idx]]
            parents_name = current_node.parents
            parents_idx = self.feature_info.get_position_by_name(parents_name)

            parents_val = fake_sample[:, parents_idx] if parents_idx != [] else None

            # --- THE TRICK: INTERVENTION ---
            # If this node has parents, we might break the link for anomaly samples
            if parents_val is not None:
                # Create a shuffled version of parents (Broken Link)
                shuffled_indices = torch.randperm(batch_size)
                broken_parents = parents_val[shuffled_indices]

                # Select: Use Real Parents for Normal, Broken Parents for Anomaly
                # We only break ONE edge per anomaly sample to make it subtle/hard
                # (Logic: randomly pick a node to break for each anomaly sample)
                # For simplicity here, we apply the shuffle logic masked by 'is_anomaly'
                # In a robust version, you ensure only 1 node is broken per row.

                # Apply the mask:
                # If is_anomaly is True, use broken_parents. Else use parents_val.
                # Note: This implementation breaks ALL dependencies for anomalies.
                # To be harder, break only specific ones.
                final_parents = torch.where(is_anomaly.unsqueeze(1), broken_parents, parents_val)
            else:
                final_parents = None

            val_position = self.feature_info.get_position_by_name(self.idx2name[idx])
            fake_sample[:, val_position] = current_node.cal_val(exogenous_var, final_parents)

        return fake_sample, labels

    def sample_latent_shift(self, batch_size, corruption_prob=0.5, shift_mean=3.0, shift_std=1.0):
        """
        Generate marginal anomalies by shifting the latent noise distribution.

        Args:
            batch_size: Number of samples to generate.
            corruption_prob: Probability of a sample being an anomaly.
            shift_mean: Mean of the noise for anomalies (Normal is 0.0).
                        Setting this to ~3.0 pushes values to the tails (e.g., Age=90).
            shift_std: Std deviation of noise for anomalies (Normal is 1.0).
                       Setting this > 1.0 increases variance (extreme low/high values).

        Returns:
            fake_sample: Generated data [batch_size, n_features]
            labels: 0 for Normal, 1 for Anomaly [batch_size]
        """
        # Initialize output containers
        fake_sample = torch.zeros((batch_size, sum(self.feature_info.dim_info.values()))).to(self.device)
        labels = torch.zeros(batch_size).to(self.device)  # 0=Normal

        # 1. Determine which samples will be anomalies
        is_anomaly = (torch.rand(batch_size) < corruption_prob).to(self.device)
        labels[is_anomaly] = 1.0

        # 2. Iterate through nodes in topological order
        for idx in range(len(self.nodes)):
            current_node = self.nodes[self.idx2name[idx]]

            # --- THE TRICK: LATENT SHIFT ---
            # Generate Standard Noise (Normal)
            z_normal = torch.randn(batch_size, self.config.z_dim, device=self.device)

            # Generate Shifted Noise (Anomaly)
            # z ~ N(shift_mean, shift_std)
            # This pushes the generator to output extreme/unseen values
            z_anomaly = torch.randn(batch_size, self.config.z_dim, device=self.device) * shift_std + shift_mean

            # Select noise based on the label
            # is_anomaly shape [B] -> [B, 1] for broadcasting
            exogenous_var = torch.where(is_anomaly.unsqueeze(1), z_anomaly, z_normal)

            # 3. Prepare Parents (Standard Causal Logic)
            parents_name = current_node.parents
            parents_idx = self.feature_info.get_position_by_name(parents_name)
            parents_val = fake_sample[:, parents_idx] if parents_idx != [] else None

            # 4. Generate Value
            # Note: We do NOT break causal links here. We only change 'z'.
            # The result is a value that respects the parent (mechanistically)
            # but is likely an extreme outlier for that context.
            val_position = self.feature_info.get_position_by_name(self.idx2name[idx])
            fake_sample[:, val_position] = current_node.cal_val(exogenous_var, parents_val)

        return fake_sample, labels

    def node_order(self):
        """
        Topology sorting: Reorder the node/feature order in dataset to the topology (from root -> leaf) order of causal graph.
        """
        check_list = []
        graph = self.causal_graph.copy()
        n_feature = len(graph)
        for node in graph:
            # nodes with no parents
            if node[1] == []:
                check_list.append(node[0])
        n_independent = len(check_list)
        n_dependent = n_feature - n_independent
        n_try = 0
        while (len(graph) != 0):
            if (graph[0][1] == []):
                graph.remove(graph[0])
                continue
            flag = 1
            for b in graph[0][1]:
                if b not in check_list:
                    flag = 0
            # all parents of the current node is in check_list
            if flag == 1:
                # print("predict this dependent variable based on its parents")
                check_list.append(graph[0][0])
                graph.remove(graph[0])
            else:
                # print("this dependent variable has some of its parents not defined yet")
                tem = graph[0]
                graph.remove(graph[0])
                graph.append(tem)
                n_try += 1
                if n_try > (n_dependent * 3):
                    print("cannot generate dependent variables")
                    check_list.append(tem[0])
                    graph.remove(graph[-1])

        name2idx = {}
        for idx, item in enumerate(check_list):
            name2idx[item] = idx
        return name2idx

    def restore_from_checkpoints(self, checkpoints):
        """ Load causal mechanisms for all nodes from checkpoints
        :param checkpoints: dict: key: node name; value: checkpoint
        """
        for k in self.nodes.keys():
            self.nodes[k].load_checkpoint(checkpoints[k])

    def fetch_checkpoints(self):
        """
        Fetch stat_dicts from causal mechanisms of all nodes
        :return: dict: key: node name; value: checkpoint
        """
        checkpoints = {}
        for k in self.nodes.keys():
            checkpoints[k] = self.nodes[k].fetch_checkpoint()
        return checkpoints

    def get_causal_mechanisms(self):
        """
        Get causal mechanisms(generator) of each nodes.
        :return:
        """
        return [node.causal_mechanism for node in self.nodes]

    def get_causal_mechanisms_params(self):
        return [{'params': self.nodes[k].causal_mechanism.parameters()} for k in self.nodes.keys()]

    def set_causal_mechanisms_train(self):
        for k in self.nodes.keys():
            self.nodes[k].causal_mechanism.train()

    def set_causal_mechanisms_eval(self):
        for k in self.nodes.keys():
            self.nodes[k].causal_mechanism.eval()

    def set_causal_mechanisms_zero_grad(self):
        for k in self.nodes.keys():
            self.nodes[k].causal_mechanism.zero_grad()

