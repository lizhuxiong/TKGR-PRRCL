# Copyright (c) Facebook, Inc. and its affiliates.

# from abc import ABC, abstractmethod
# from re import L
from typing import Tuple, List, Dict
from neibs_info_embedding3 import Path
from contrastive_learning5 import ConLoss_his
from his_embdding import EventEmbeddingModel
import torch
from torch import nn
import torch.nn.functional as F


class PRRCL(nn.Module):
    def __init__(
            self, args, sizes: Tuple[int, int, int, int], rank: int,
            is_cuda: bool = False
    ):
        super(PRRCL, self).__init__()
        self.model_name = "PRRCL"
        self.sizes = sizes
        self.rank = rank  # dimension
        self.embeddings = nn.ModuleList([
            nn.Embedding(sizes[0], rank, sparse=True),  # 0 entityies (7128,1200*2)
            nn.Embedding(sizes[1], rank, sparse=True),  # 1 relaitions
            nn.Embedding(sizes[3], rank, sparse=True),  # 2 entity timstamps
            nn.Embedding(sizes[3], rank, sparse=True),  # 3 relation timstamps
            nn.Embedding(sizes[0], rank, sparse=True),  # 4 his

        ])
        self.is_cuda = is_cuda

        feature_length = rank
        self.pathNet_encoder = Path(feature_length, args).to(args.cuda)
        self.contrastive_leanring_his = ConLoss_his(args).to(args.cuda)

        self.his_emb_model = EventEmbeddingModel(args.cuda, rank, rank)

        if rank % 2 != 0:
            raise "rank need to be devided by 2.."
        for i in range(len(self.embeddings)):
            torch.nn.init.xavier_uniform_(self.embeddings[i].weight.data)

    @staticmethod
    def has_time():
        return False

    def mul(self, emb1, emb2):
        a, b = torch.chunk(emb1, 2, dim=1)
        c, d = torch.chunk(emb2, 2, dim=1)
        return torch.cat([(a * c - b * d), (a * d + b * c)], dim=1)

    def forward_over_time(self, x):
        raise NotImplementedError("no.")

    def complex_mul(self, emb1, emb2):
        a, b = torch.chunk(emb1, 2, dim=1)
        c, d = torch.chunk(emb2, 2, dim=1)
        return torch.cat(((a * c + b * d), (a * d - b * c)), dim=1)  # 沿列拼接  #复数公式计算错误？ 没有，其实不是复数计算，就是元素乘积

    def forward(self, args, x, batch_history, mode, neis_all, neis_timestamps, path_weight, epoch,
                filters: Dict[Tuple[int, int, int], List[int]] = None):
        # get the his_emb

        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])

        all_en = self.embeddings[0].weight.transpose(0, 1)
        en_his = self.embeddings[4](x[:, 0])

        time_ent = self.embeddings[2](x[:, 3])
        time_rel = self.embeddings[3](x[:, 3])

        neis_embd = self.forward_PathNet(self.pathNet_encoder, self.embeddings[0].weight, self.embeddings[2].weight,
                                         args.num_walks, args.walk_len,
                                         neis_all, neis_timestamps, path_weight, x[:, 0], x[:, 3])

        en_his_ = self.mul(neis_embd, en_his)
        lhs2 = neis_embd + self.mul(time_ent, en_his_)
        lhs_ = lhs + lhs2

        if mode == 'Training':
            # get the contrastive_leanring_loss
            contrastive_leanring_loss = self.contrastive_leanring_his(x[:, 1], self.embeddings[1].weight, time_rel, batch_history)  # rel
            lhs_rel = self.mul(torch.cat([rel, time_rel], dim=1),
                               torch.cat([lhs_, time_ent], dim=1))

            # lhs_rel = F.dropout(lhs_rel, p=0.1, training=self.training)
            a, b = torch.chunk(lhs_rel, chunks=2, dim=1)
            scores = a @ all_en + torch.sum(b * neis_embd, dim=1).unsqueeze(1)
            return scores, contrastive_leanring_loss

        else:
            with torch.no_grad():
                lhs_rel = self.mul(torch.cat([rel, time_rel], dim=1),
                                   torch.cat([lhs_, time_ent], dim=1))
                a, b = torch.chunk(lhs_rel, chunks=2, dim=1)
                scores = a @ all_en + torch.sum(b * neis_embd, dim=1).unsqueeze(1)
                targets = []
                for (score, target_index) in zip(scores, x[:, 2]):
                    targets.append(score[target_index])
                targets = torch.tensor(targets).view(-1, 1).to(args.cuda)
                # set filtered and true scores to -1e6 to be ignored
                for i, query in enumerate(x):
                    filter_out = filters[(query[0].item(), query[1].item(), query[3].item())]  # 不预测同步记录
                    filter_out += [x[i, 2].item()]
                    scores[i, torch.LongTensor(filter_out)] = -1e6

            return scores, targets

    def forward_PathNet(self, pathNet, entity_embedding, en_time_embedding, num_walks, walk_len,
                        neis_all, neis_timestamps, path_weight,
                        entitys, cur_time):
        neis = neis_all[entitys]
        neis_time = neis_timestamps[entitys]
        weight = path_weight[entitys]

        for index, time in enumerate(neis_time):
            if time[:1].item() == -1:  # Handle nodes without paths  [13, 13, 13, -1, -1, -1, 0, 0, 0]  -> [13, 13, 13, en_time, en_time, en_time, 1, 1, 1]
                neis_time[index].fill_(cur_time[index])
                # weight[index].fill_(1)
        # pathNet.train()
        # entity_embedding shape(7129, 200)
        neibs_info_embedding = pathNet(entity_embedding, en_time_embedding, neis, num_walks, walk_len, neis_time,
                                       weight)
        return neibs_info_embedding  # shape(the number of all nodes , hidden_size) ==> shape(the number of current epoch, hidden_size)=(neis.shape[0], hidden_size)
