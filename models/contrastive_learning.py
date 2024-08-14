import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class ConLoss_his(nn.Module):

    def __init__(self, args, temperature = 1, scale_by_temperature=True):
        super(ConLoss_his, self).__init__()
        self.device = args.cuda
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature

    def forward(self, entities, entity_embedding, time_ent, history):
        features = entity_embedding
        # set_entities = list(set(entities.tolist()))
        # num_dif = len(set_entities)
        list_entities = entities.tolist()
        set_entities = []
        set_en_time = []
        his_entities, his_time = [], []
        for index, en in enumerate(list_entities):
            if en not in set_entities:
                set_entities.append(en)
                set_en_time.append(time_ent[index])
                # his_en, his_time = self.batch_his_e_t(history[index])
                his_entities.append(history[index])
        set_en_time = torch.stack(set_en_time).to(self.device)
        num_dif = len(set_entities)
        similar_entities = []  # get the samilar entities ids. shape(num_dif, )
        for en1 in his_entities:
            similar_ = []
            if (len(en1) != 0):
                for index, en2 in enumerate(his_entities):
                    num_intersection = len(set(en1).intersection(set(en2)))
                    if (num_intersection >= 2):
                        similar_.append((index, num_intersection))
            similar_entities.append(similar_)

        features_dim = entity_embedding.shape[1]
        batch_embed = torch.zeros(num_dif, features_dim).to(self.device)
        for index, entity in enumerate(set_entities):
            batch_embed[index] = features[entity]

        batch_embed = torch.cat([batch_embed, set_en_time], dim=1)

        # batch_embed = F.normalize(batch_embed, p=2, dim=1)
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(batch_embed, batch_embed.T),
            self.temperature)
        # for numerical stability

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        exp_logits = torch.exp(logits)

        # make mask
        positives_mask = torch.zeros(num_dif, num_dif, dtype=torch.float32).to(self.device)

        for index in range(num_dif):
            similar_en = similar_entities[index]
            if (len(similar_en) == 0):
                continue
            for i, _ in similar_en:
                if index != i:
                    positives_mask[index][i] = 1
                    positives_mask[i][index] = 1
        # negatives_mask = 1 - positives_mask
        # for index in range(num_dif):
        #     negatives_mask[index][index] = 0

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # mask_pos_pairs = positives_mask.sum(1)
        # mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)

        count = (positives_mask > 0).sum(1)
        count = torch.where(count < 1e-6, 1, count)
        log_probs = (positives_mask * log_prob).sum(1) / count

        # loss
        loss = -log_probs
        if self.scale_by_temperature:
            loss *= self.temperature
        loss = loss.mean()
        return loss

    def batch_his_e_t(self, his):
        entity, time = [], []
        for en, tim in his:
            # entity.append(int(en))
            # time.append(int(tim))
            entity.append(en)
            time.append(tim)
        # return torch.tensor(entity).to(self.device), torch.tensor(time).to(self.device)
        return entity, time

