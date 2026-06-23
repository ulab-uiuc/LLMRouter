from itertools import combinations

import torch
import torch.nn.functional as F
from torch_geometric.nn import GeneralConv
from torch_geometric.data import Data
import torch.nn as nn
from torch.optim import AdamW
from sklearn.metrics import f1_score

class FeatureAlign(nn.Module):

    def __init__(self, query_feature_dim, llm_feature_dim, user_feature_dim, common_dim):
        super(FeatureAlign, self).__init__()
        self.user_transform = nn.Linear(user_feature_dim, common_dim)
        self.query_transform = nn.Linear(query_feature_dim, common_dim)
        self.llm_transform = nn.Linear(llm_feature_dim, common_dim*3)
        self.task_transform = nn.Linear(llm_feature_dim, common_dim)

    def forward(self,task_id, query_features, llm_features, user_features):
        aligned_task_features = self.task_transform(task_id)
        aligned_query_features = self.query_transform(query_features)
        aligned_user_features = self.user_transform(user_features)
        aligned_user_features = aligned_user_features.repeat(int(aligned_task_features.size(0)/aligned_user_features.size(0))+1, 1)
        aligned_user_features = aligned_user_features[:aligned_task_features.size(0)]
        aligned_three_features=torch.cat([aligned_task_features,aligned_query_features,aligned_user_features], 1)
        aligned_llm_features = self.llm_transform(llm_features)
        aligned_features = torch.cat([aligned_three_features, aligned_llm_features], 0)
        return aligned_features


class EncoderDecoderNet(torch.nn.Module):

    def __init__(self, query_feature_dim, llm_feature_dim, user_feature_dim, hidden_features, in_edges):
        super(EncoderDecoderNet, self).__init__()
        self.in_edges = in_edges
        self.model_align = FeatureAlign(query_feature_dim, llm_feature_dim, user_feature_dim, hidden_features)
        self.encoder_conv_1 = GeneralConv(in_channels=hidden_features* 3, out_channels=hidden_features* 3, in_edge_channels=in_edges)
        self.encoder_conv_2 = GeneralConv(in_channels=hidden_features* 3, out_channels=hidden_features* 3, in_edge_channels=in_edges)

        self.edge_mlp = nn.Linear(in_edges, in_edges)
        self.bn1 = nn.BatchNorm1d(hidden_features * 3)
        self.bn2 = nn.BatchNorm1d(hidden_features * 3)

        self.edge_decoder = nn.Sequential(
            nn.Linear(hidden_features * 3 * 4, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, 1)
        )

    def forward(self, task_id, query_features, llm_features, user_features, edge_index, edge_mask=None,
                edge_can_see=None, edge_weight=None):
        if edge_mask is not None:
            num_edge = edge_mask.size(0)
            edge_index_part1 = edge_index[:, :num_edge]
            edge_index_part2 = edge_index[:, num_edge:]   #llm-llm edge
            edge_index_part1_mask = edge_index_part1[:, edge_can_see]
            edge_index_mask = torch.cat([edge_index_part1_mask, edge_index_part2], dim=1)
            edge_index_predict = edge_index_part1[:, edge_mask]
            if edge_weight is not None:
                num_llm_llm_edge = edge_index_part2.size(1)
                edge_weight_mask = edge_weight[edge_can_see]
                llm_llm_edge_weight_mask = torch.ones((num_llm_llm_edge, self.in_edges),device=edge_weight_mask.device)
                edge_weight_mask = torch.cat([edge_weight_mask, llm_llm_edge_weight_mask], dim=0)
        edge_weight_mask=F.relu(self.edge_mlp(edge_weight_mask.reshape(-1, self.in_edges)))
        edge_weight_mask = edge_weight_mask.reshape(-1,self.in_edges)
        x_ini = (self.model_align(task_id, query_features, llm_features, user_features))
        x = F.relu(self.bn1(self.encoder_conv_1(x_ini, edge_index_mask, edge_attr=edge_weight_mask)))
        x = self.bn2(self.encoder_conv_2(x, edge_index_mask, edge_attr=edge_weight_mask))

        edge_predict = F.sigmoid(
            (x_ini[edge_index_predict[0]] * x[edge_index_predict[1]]).mean(dim=-1))

        return edge_predict

class form_data:

    def __init__(self,device):
        self.device = device

    def classify_llm_family(self,llm_mapping,config):
        llm_family = config['llm_family']
        result = {}
        for family in llm_family:
            lower_model = family.lower()
            indices = []
            for key, idx in llm_mapping.items():
                if isinstance(key, str) and lower_model in key.lower():
                    indices.append(idx)
            result[family] = indices if indices else []
        return result

    def add_llm_edges(self,llm_family_indices, edge_index,start_idx):
        new_edges = []
        for indices in llm_family_indices.values():
            if len(indices) >= 2:
                for u, v in combinations(indices, 2):
                    new_edges.append([u+start_idx, v+start_idx])
                    new_edges.append([v+start_idx, u+start_idx])

        if not new_edges:
            return edge_index
        new_edges_tensor = torch.tensor(new_edges, dtype=torch.long, device=edge_index.device).T
        edge_index = torch.cat([edge_index, new_edges_tensor], dim=1)
        return edge_index

    def formulation(self,task_id,query_feature,llm_feature,user_feature,org_node,des_node,edge_feature,label,edge_mask,combined_edge,train_mask,valide_mask,test_mask,llm_mapping,best_llm,cost_list,config):

        query_features = torch.tensor(query_feature, dtype=torch.float).to(self.device)
        llm_features = torch.tensor(llm_feature, dtype=torch.float).to(self.device)
        user_features = torch.tensor(user_feature, dtype=torch.float).to(self.device)
        task_id=torch.tensor(task_id, dtype=torch.float).to(self.device)
        cost_list = torch.tensor(cost_list, dtype=torch.float).to(self.device)
        llm_family_indices = self.classify_llm_family(llm_mapping,config)
        query_indices = list(range(len(query_features)))
        llm_indices = [i + len(query_indices) for i in range(len(llm_features))]
        des_node=[(i + query_feature.shape[0]) for i in des_node]
        edge_index = torch.tensor([org_node, des_node], dtype=torch.long).to(self.device)
        edge_index = self.add_llm_edges(llm_family_indices,edge_index,start_idx=query_feature.shape[0])
        edge_weight = torch.tensor(edge_feature, dtype=torch.float).reshape(-1,1).to(self.device)
        combined_edge = edge_weight
        data = Data(task_id=task_id,query_features=query_features, llm_features=llm_features, user_features=user_features, edge_index=edge_index,
                        edge_attr=edge_weight,query_indices=query_indices, llm_indices=llm_indices,label=torch.tensor(label, dtype=torch.float, device=self.device),edge_mask=edge_mask,combined_edge=combined_edge,best_llm=best_llm,
                    train_mask=train_mask,valide_mask=valide_mask,test_mask=test_mask,cost_list=cost_list)

        return data


class GNN_prediction:
    def __init__(self, query_feature_dim, llm_feature_dim,user_feature_dim,hidden_features_size,in_edges_size,config,device):

        self.model = EncoderDecoderNet(query_feature_dim=query_feature_dim, llm_feature_dim=llm_feature_dim,user_feature_dim=user_feature_dim,
                                        hidden_features=hidden_features_size,in_edges=in_edges_size).to(device)
        self.config = config
        self.optimizer =AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2.0).to(device))

    def train_validate(self,data,data_validate,data_for_test):
        best_test_result = 0.0
        best_f1=-1
        self.save_path= self.config['model_path']
        self.num_edges = len(data.edge_attr)
        self.train_mask = torch.tensor(data.train_mask, dtype=torch.bool)
        self.valide_mask = torch.tensor(data.valide_mask, dtype=torch.bool)
        self.test_mask = torch.tensor(data.test_mask, dtype=torch.bool)
        for epoch in range(self.config['train_epoch']):
            self.model.train()
            loss_mean=0
            mask_train = data.edge_mask
            for inter in range(self.config['batch_size']):
                mask = mask_train.clone()
                mask = mask.bool()
                random_mask = torch.rand(mask.size()) < self.config['train_mask_rate']
                random_mask = random_mask.to(torch.bool)
                mask = torch.where(mask & random_mask, torch.tensor(False, dtype=torch.bool), mask)
                mask = mask.bool()

                #-------large scale setting----------
                # two_drop_prob = self.config["two_drop_prob"]
                # seven_drop_prob = self.config["seven_drop_prob"]
                #
                # n = mask.size(0)
                # idx_tensor = torch.arange(n, device=mask.device)
                #
                # two_mask_condition = (idx_tensor % 10 == 2) & mask
                # two_random_mask = torch.rand(n, device=mask.device) < two_drop_prob
                # seven_mask_condition = (idx_tensor % 10 == 7) & mask
                # seven_random_mask = torch.rand(n, device=mask.device) < seven_drop_prob
                # mask[two_mask_condition & two_random_mask] = False
                # mask[seven_mask_condition & seven_random_mask] = False

                edge_can_see = (~mask) & self.train_mask

                self.optimizer.zero_grad()
                predicted_edges= self.model(task_id=data.task_id,query_features=data.query_features, llm_features=data.llm_features, user_features=data.user_features, edge_index=data.edge_index,
                                            edge_mask=mask,edge_can_see=edge_can_see,edge_weight=data.combined_edge)
                loss = self.criterion(predicted_edges.reshape(-1), data.label[mask].reshape(-1))
                loss_mean+=loss
            loss_mean=loss_mean/self.config['batch_size']
            loss_mean.backward()
            self.optimizer.step()
            self.model.eval()
            mask_validate = data_validate.edge_mask.clone().detach().bool()
            edge_can_see = self.train_mask
            with torch.no_grad():
                predicted_edges_validate = self.model(task_id=data_validate.task_id,query_features=data_validate.query_features,
                                                                            llm_features=data_validate.llm_features,
                                                                            user_features=data_validate.user_features,
                                                                            edge_index=data_validate.edge_index,
                                                                            edge_mask=mask_validate,edge_can_see=edge_can_see, edge_weight=data_validate.combined_edge)
                observe_edge= predicted_edges_validate.reshape(self.config['user_num'], -1, self.config['llm_num'])
                observe_idx = torch.argmax(observe_edge, 2)
                value_validate=data_validate.edge_attr[mask_validate].reshape(self.config['user_num'], -1, self.config['llm_num'])
                label_idx = torch.argmax(value_validate, 2)
                correct = (observe_idx == label_idx).sum().item()
                total = label_idx.size(0)
                validate_accuracy = correct / total
                observe_idx_ = observe_idx.cpu().numpy().reshape(-1,1)
                label_idx_ = label_idx.cpu().numpy().reshape(-1,1)
                f1 = f1_score(label_idx_, observe_idx_, average='macro')
                loss_validate = self.criterion(predicted_edges_validate.reshape(-1), data_validate.label[mask_validate].reshape(-1))

                # Model selection is driven by validation f1 only. When validation
                # improves we checkpoint the model and record THAT checkpoint's test
                # score as the reported result. We intentionally do NOT take a max()
                # over per-epoch test scores: doing so selects the best epoch by
                # peeking at the test set, producing an optimistic, leakage-biased
                # number that also wouldn't match the saved (best-validation) model.
                improved = f1 >= best_f1
                if improved:
                    best_f1 = f1
                    torch.save(self.model.state_dict(), self.save_path)
                    test_result, test_loss = self.test(data_for_test, self.config['model_path'])
                    best_test_result = test_result

                print(
                    f"[PersonalizedRouter] epoch {epoch + 1}/{self.config['train_epoch']} | "
                    f"val_acc={validate_accuracy:.4f} val_loss={loss_validate.item():.4f} "
                    f"val_f1={f1:.4f} | best_val_f1={best_f1:.4f} test@best_val={best_test_result:.4f}"
                    + (" *" if improved else "")
                )
        self.best_test_result = best_test_result

    def test(self,data,model_path):
        # state_dict = torch.load(model_path)
        # self.model.load_state_dict(state_dict)
        self.model.eval()
        mask = data.edge_mask.clone().detach().bool()
        # self.valide_mask = torch.tensor(data.valide_mask, dtype=torch.bool)
        # self.train_mask = torch.tensor(data.train_mask, dtype=torch.bool)
        edge_can_see = torch.logical_or(self.valide_mask, self.train_mask)
        with torch.no_grad():
            edge_predict = self.model(task_id=data.task_id,query_features=data.query_features, llm_features=data.llm_features, user_features=data.user_features,edge_index=data.edge_index,
                             edge_mask=mask,edge_can_see=edge_can_see,edge_weight=data.combined_edge)
        label = data.label[mask].reshape(-1)
        loss_test = self.criterion(edge_predict, label)
        edge_predict = edge_predict.reshape(-1, self.config['llm_num'])
        max_idx = torch.argmax(edge_predict, 1)
        value_test = data.edge_attr[mask].reshape(-1, self.config['llm_num'])
        label_idx = torch.argmax(value_test, 1)
        row_indices = torch.arange(len(value_test))
        result = value_test[row_indices, max_idx].mean()
        result_golden = value_test[row_indices, label_idx].mean()
        print("result_predict:", result, "result_golden:",result_golden)

        return result,loss_test