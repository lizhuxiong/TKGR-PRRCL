import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class EventEmbeddingModel(nn.Module):
    def __init__(self, device, input_dim_m, output_dim_sm):
        super(EventEmbeddingModel, self).__init__()
        self.dim = output_dim_sm
        self.LinearQ = nn.Linear(input_dim_m, output_dim_sm)
        self.device = device

    def forward(self, entities, history, entities_emb):
        # entities_his_Nor = F.normalize(entities_his, p=2, dim=1)
        # event_times_nor = F.normalize(event_times, p=2, dim=1)
        his_tim_embedding = [] #batch_size * dimension ,get the batch entities&time history info
        entities = entities.tolist()
        for event_his, ent in zip(history, entities):
            if len(event_his) == 0: # no his
                his_1 = entities_emb[ent].view(1,-1)
            else:
                entity_embed=torch.zeros(1, self.dim).to(self.device)
                for his_entity in event_his:
                    entity_embed = entity_embed + entities_emb[his_entity]
                his_1 = torch.mean(entity_embed, dim=0).view(1,-1) #way 1
            his_tim_embedding.append(self.LinearQ(his_1))
        his_time_embedding_1 = torch.cat(his_tim_embedding, dim=0)
        return his_time_embedding_1

    # get current evernt history
    def batch_his_e_t(self, his):
        entity, time = [], []
        for en, tim in his:
            # entity.append(int(en))
            # time.append(int(tim))
            entity.append(en)
            time.append(tim)
        return torch.tensor(entity).to(self.device), torch.tensor(time).to(self.device)

    # Define an exponential time decay function
    def time_decay_weight(self, delta_t, decay_rate = torch.tensor(1)):
        decay_rate = decay_rate.to(self.device)
        delta = torch.exp(-decay_rate * delta_t)
        return delta
    
