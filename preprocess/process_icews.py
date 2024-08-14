import os
from pathlib import Path
import pickle

import numpy as np
import tqdm
from collections import defaultdict

DATA_PATH = "../data/"

def prepare_dataset(path, name):
    files = ['train', 'valid', 'test']
    entities, relations, timestamps = set(), set(), set()
    for f in files:
        file_path = os.path.join(path, f)
        to_read = open(file_path, 'r')
        for line in to_read.readlines():
            lhs, rel, rhs, timestamp = line.strip().split('\t')
            # lhs, rel, rhs, timestamp, _ = line.replace(' ', '').split('\t') #ICEWS18 、GDELT
            entities.add(lhs)
            entities.add(rhs)
            relations.add(rel)
            timestamps.add(timestamp)
        to_read.close()

    entities_to_id = {x: i for (i, x) in enumerate(sorted(entities))}
    relations_to_id = {x: i for (i, x) in enumerate(sorted(relations))}
    timestamps_to_id = {x: i for (i, x) in enumerate(sorted(timestamps))}

    print("{} entities, {} relations over {} timestamps".format(len(entities), len(relations), len(timestamps)))
    stat = []
    stat.append(len(entities))
    n_relations = len(relations)
    stat.append(n_relations)
    stat.append(len(timestamps))
    print(stat)
    out_stat = open(Path(DATA_PATH) / name / ('stat'), 'wb')
    pickle.dump(np.array(stat), out_stat)
    out_stat.close()

    # os.makedirs(os.path.join(DATA_PATH, name))
    # write ent to id / rel to id
    for (dic, f) in zip([entities_to_id, relations_to_id, timestamps_to_id], ['ent2id.txt', 'rel2id.txt', 'ts2id.txt']):
        ff = open(os.path.join(DATA_PATH, name, f), 'w+')
        for (x, i) in dic.items():
            ff.write("{}\t{}\n".format(x, i))
        ff.close()

    # map train/test/valid with the ids
    for f in files:
        file_path = os.path.join(path, f)
        to_read = open(file_path, 'r')
        examples = []
        for line in to_read.readlines():
            lhs, rel, rhs, timestamp = line.strip().split('\t')
            try:
                examples.append([entities_to_id[lhs], relations_to_id[rel], entities_to_id[rhs], timestamps_to_id[timestamp]])
            except ValueError:
                continue
        examples = sorted(examples, key=lambda x: x[3])

        out = open(Path(DATA_PATH) / name / (f + '.pickle'), 'wb')
        pickle.dump(np.array(examples), out)
        out.close()

    print("creating filtering lists")

    # create filtering files
    to_skip = {'lhs': defaultdict(set), 'rhs': defaultdict(set)}
    # for f in files:
    for f in ['valid', 'test']:
        examples = pickle.load(open(Path(DATA_PATH) / name / (f + '.pickle'), 'rb'))
        for lhs, rel, rhs, ts in examples:
            to_skip['lhs'][(rhs, rel + n_relations, ts)].add(lhs)  # reciprocals
            to_skip['rhs'][(lhs, rel, ts)].add(rhs)

    to_skip_final = {'lhs': {}, 'rhs': {}}
    for kk, skip in to_skip.items():
        for k, v in skip.items():
            to_skip_final[kk][k] = sorted(list(v))

    out = open(Path(DATA_PATH) / name / 'to_skip.pickle', 'wb')
    pickle.dump(to_skip_final, out)
    out.close()

    # get train synchronization
    to_skip2 = {'lhs': defaultdict(set), 'rhs': defaultdict(set)}
    samples2 = pickle.load(open(Path(DATA_PATH) / name / ('train.pickle'), 'rb'))
    for lhs, rel, rhs, ts in samples2:
        to_skip2['lhs'][(rhs, rel + n_relations, ts)].add(lhs)  # reciprocals
        to_skip2['rhs'][(lhs, rel, ts)].add(rhs)
    train_syncro = {'lhs': {}, 'rhs': {}}
    for kk, skip in to_skip2.items():
        for k, v in skip.items():
            train_syncro[kk][k] = sorted(list(v))
    out = open(Path(DATA_PATH) / name / 'train_syncro.pickle', 'wb')
    pickle.dump(train_syncro, out)
    out.close()

    # def get_event_history():
    # defaultdict(set) -> list，so that can use index
    en_rel = {'lhs': [], 'rhs': []}
    en_rel_synchr = {'lhs': [], 'rhs': []}
    for kk, skip in train_syncro.items():
        for k, v in skip.items():
            en_rel[kk].append(k) #current entity and relation
            en_rel_synchr[kk].append(v) #

    history_events = {'lhs': {}, 'rhs': {}}  # 'lhs': {[()]}
    len_en_rel = len(en_rel['lhs']) #
    len_en_rel2 = len(en_rel['rhs']) # ********note****** ---------len_en_rel donnot equal len_en_rel2----------
    # # print(len_en_rel)


    for kk in ['lhs', 'rhs']:
        # skip = en_rel[kk]
        for (entity_cur, rel_cur, time_cur), i in zip(en_rel[kk][:], tqdm.tqdm(range(0, len_en_rel) if kk == 'lhs' else range(0, len_en_rel2))):
            if(time_cur == 0):
                history_events[kk][(entity_cur, rel_cur, time_cur)] = []
            else:
                # history_events[kk][(entity_c, rel_c, time_c)] = en_time[kk][i]
                for num, (entity_former, rel_former, time_former) in enumerate(en_rel[kk][i-1::-1]):
                    index_cur = i - 1 - num
                    if(entity_cur == entity_former and rel_cur == rel_former):
                        if(time_cur != time_former):
                            history_events[kk][(entity_cur, rel_cur, time_cur)] = en_rel_synchr[kk][index_cur]
                            history_events[kk][(entity_cur, rel_cur, time_cur)].extend(history_events[kk][(entity_former, rel_former, time_former)]) #1) second: add in front of the previous(time-n-1) history
                        break
                    if(index_cur == 0):
                        history_events[kk][(entity_cur, rel_cur, time_cur)] = []
    out_history = open(Path(DATA_PATH) / name / 'history.pickle', 'wb')
    pickle.dump(history_events, out_history)
    out.close()

if __name__ == "__main__":
    # datasets = ['ICEWS14', "ICEWS05-15",'GDELT',"ICEWS18"]
    datasets = ['ICEWS14']
    for d in datasets:
        print("Preparing dataset {}".format(d))
        prepare_dataset(os.path.join(DATA_PATH, d),d)


