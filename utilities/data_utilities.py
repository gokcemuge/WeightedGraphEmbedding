# encoding: utf-8
import os
import pandas as pd
import numpy as np
import random
from sklearn.utils import shuffle as skshuffle

class KnowledgeGraph:

    def __init__(self, data_dir, train_with_groundings=False, train_with_psl=False):
        self.data_dir = data_dir
        self.dataset_name = data_dir.split("/")[-1].lower()

        self.train_with_groundings = train_with_groundings
        self.train_with_psl = train_with_psl

        self.entities = []  # entity list
        self.id2ent_dict = {}  # id to entity dict
        self.ent2id_dict = {}  # entity to id dict--needed for nl27k

        self.id2rel_dict = {}  # id to rel dict
        self.rel2id_dict = {}  # rel to id dict--needed for nl27k

        self.n_entity = 0
        self.n_relation = 0

        self.training_triples = []  # form of (h, t, r , w)
        self.validation_triples = []  # form of (h, t, r , w)
        self.test_triples = []  # form of (h, t, r , w)

        self.n_training_triple = 0
        self.n_validation_triple = 0
        self.n_test_triple = 0

        if train_with_psl:
            self.psl_triples = []  # form of (h, t, r , w)
            self.n_psl_triple = 0
        if train_with_groundings:
            self.groundings = []  # form of (h, t, r , w)
            self.n_groundings = 0

        self.hr_map = {}
        self.hr_map_sub = {}

        # stores all positive triples
        self.all_pos_triples = {}
        self.all_pos_but_test_triples = {}
        self.train_triple_filter = {}
        self.train_val_triple_filter = {}
        self.train_val_test_triple_filter = {}

        # for triple specific slack variables
        self.triple2idx = {}
        self.rel2idx = {}

        '''load dicts and triples'''
        self.load_dicts()
        self.load_triples()
        # TODO: hr map for ranking
        self.load_hr_map(self.test_triples, self.training_triples, self.validation_triples)

    def load_dicts(self):
        entity_dict_file = 'entity_id.csv'
        relation_dict_file = 'relation_id.csv'

        if self.dataset_name == 'nl27k' or self.dataset_name == 'sch40k' or self.dataset_name == 'aida35k':  # (id, entity)

            print('-----Loading entity dict-----')
            try:
                entity_df = pd.read_csv(os.path.join(str(self.data_dir), entity_dict_file), header=None)
                self.id2ent_dict = dict(zip(entity_df[0], entity_df[1]))
                self.ent2id_dict = dict(zip(entity_df[1], entity_df[0]))
                self.n_entity = len(self.id2ent_dict)
                self.entities = list(self.id2ent_dict.values())
                print('#entity: {}'.format(self.n_entity))
            except IOError as e:
                print(e)
                print(" Entity file not found ! ")

            print('-----Loading relation dict-----')
            try:
                relation_df = pd.read_csv(os.path.join(self.data_dir, relation_dict_file), header=None)
                self.id2rel_dict = dict(zip(relation_df[0], relation_df[1]))
                self.rel2id_dict = dict(zip(relation_df[1], relation_df[0]))
                self.n_relation = len(self.id2rel_dict)
                print('#relation: {}'.format(self.n_relation))
            except IOError as e:
                print(e)
                print(" Relation file not found ! ")

        elif self.dataset_name == 'cn15k' or self.dataset_name == 'ppi5k':  # (entity, id)

            print('-----Loading entity dict-----')
            try:
                entity_df = pd.read_csv(os.path.join(str(self.data_dir), entity_dict_file), header=None)
                self.id2ent_dict = dict(zip(entity_df[1], entity_df[0]))
                self.n_entity = len(self.id2ent_dict)
                self.entities = list(self.id2ent_dict.values())
                print('#entity: {}'.format(self.n_entity))
            except IOError as e:
                print(e)
                print(" Entity file not found ! ")

            print('-----Loading relation dict-----')
            try:
                relation_df = pd.read_csv(os.path.join(self.data_dir, relation_dict_file), header=None)
                self.id2rel_dict = dict(zip(relation_df[1], relation_df[0]))
                self.n_relation = len(self.id2rel_dict)
                print('#relation: {}'.format(self.n_relation))
            except IOError as e:
                print(e)
                print(" Relation file not found ! ")

    def load_triples(self):

        training_file = 'train.tsv'
        validation_file = 'val.tsv'
        test_file = 'test.tsv'
        psl_file = 'softlogic.tsv'
        groundings_file = 'groundings_all.tsv'

        if self.dataset_name == 'nl27k' or self.dataset_name == 'sch40k' or self.dataset_name == 'aida35k':  # data set has string of triples

            print('-----Loading training triples-----')
            training_df = pd.read_csv(os.path.join(self.data_dir, training_file), header=None, sep='\t')
            self.training_triples = list(zip([self.ent2id_dict[h] for h in training_df[0]],
                                             [self.ent2id_dict[str(t)] for t in training_df[2]],
                                             [self.rel2id_dict[r] for r in training_df[1]],
                                             [w for w in training_df[3]]
                                             ))

            self.n_training_triple = len(self.training_triples)
            print('#training triples size: {}'.format(self.n_training_triple))

            print('-----Loading validation triples-----')
            validation_df = pd.read_csv(os.path.join(self.data_dir, validation_file), header=None, sep='\t')
            self.validation_triples = list(zip([self.ent2id_dict[h] for h in validation_df[0]],
                                               [self.ent2id_dict[t] for t in validation_df[2]],
                                               [self.rel2id_dict[r] for r in validation_df[1]],
                                               [w for w in validation_df[3]]
                                               ))

            self.n_validation_triple = len(self.validation_triples)
            print('#validation triples size: {}'.format(self.n_validation_triple))

            print('-----Loading test triples------')
            test_df = pd.read_csv(os.path.join(self.data_dir, test_file), header=None, sep='\t')
            self.test_triples = list(zip([self.ent2id_dict[h] for h in test_df[0]],
                                         [self.ent2id_dict[str(t)] for t in test_df[2]],
                                         [self.rel2id_dict[r] for r in test_df[1]],
                                         [w for w in test_df[3]]
                                         ))

            self.n_test_triple = len(self.test_triples)
            print('#test triples size: {}'.format(self.n_test_triple))

            if self.train_with_psl:
                print('-----Loading psl triples------')
                psl_df = pd.read_csv(os.path.join(self.data_dir, psl_file), header=None, sep='\t')
                self.psl_triples = list(zip([self.ent2id_dict[h] for h in psl_df[0]],
                                            [self.ent2id_dict[t] for t in psl_df[2]],
                                            [self.rel2id_dict[r] for r in psl_df[1]],
                                            [w for w in psl_df[3]],
                                            ))
                self.n_psl_triple = len(self.psl_triples)
                print('#psl triples size: {}'.format(self.n_psl_triple))

            if self.train_with_groundings:
                print('-----Loading grounded triples------')
                grounding_df = pd.read_csv(os.path.join(self.data_dir, groundings_file), header=None, sep='\t')
                self.groundings = list(zip([self.ent2id_dict[h] for h in grounding_df[0]],
                                           [self.ent2id_dict[t] for t in grounding_df[2]],
                                           [self.rel2id_dict[r] for r in grounding_df[1]],
                                           [w for w in grounding_df[3]],
                                           ))
                self.n_groundings = len(self.groundings)
                print('grounded triples size: {}'.format(self.n_groundings))

            print('------------------------------------')
            print(' ')

        else:

            print('-----Loading training triples-----')
            training_df = pd.read_csv(os.path.join(self.data_dir, training_file), header=None, sep='\t')
            self.training_triples = list(zip([h for h in training_df[0]],
                                             [t for t in training_df[2]],
                                             [r for r in training_df[1]],
                                             [w for w in training_df[3]]
                                             ))

            self.n_training_triple = len(self.training_triples)
            print('#training triples size: {}'.format(self.n_training_triple))

            print('-----Loading validation triples-----')
            validation_df = pd.read_csv(os.path.join(self.data_dir, validation_file), header=None, sep='\t')
            self.validation_triples = list(zip([h for h in validation_df[0]],
                                               [t for t in validation_df[2]],
                                               [r for r in validation_df[1]],
                                               [w for w in validation_df[3]]
                                               ))

            self.n_validation_triple = len(self.validation_triples)
            print('#validation triples size: {}'.format(self.n_validation_triple))

            print('-----Loading test triples------')
            test_df = pd.read_csv(os.path.join(self.data_dir, test_file), header=None, sep='\t')
            self.test_triples = list(zip([h for h in test_df[0]],
                                         [t for t in test_df[2]],
                                         [r for r in test_df[1]],
                                         [w for w in test_df[3]]
                                         ))

            self.n_test_triple = len(self.test_triples)
            print('#test triples size: {}'.format(self.n_test_triple))

            if self.train_with_psl:
                print('-----Loading psl triples------')
                psl_df = pd.read_csv(os.path.join(self.data_dir, psl_file), header=None, sep='\t')
                self.psl_triples = list(zip([h for h in psl_df[0]],
                                            [t for t in psl_df[2]],
                                            [r for r in psl_df[1]],
                                            [w for w in psl_df[3]],
                                            ))
                self.n_psl_triple = len(self.psl_triples)
                print('#psl triples size: {}'.format(self.n_psl_triple))

            if self.train_with_groundings:
                print('-----Loading grounded triples------')
                grounding_df = pd.read_csv(os.path.join(self.data_dir, groundings_file), header=None, sep='\t')
                self.groundings = list(zip([h for h in grounding_df[0]],
                                           [t for t in grounding_df[2]],
                                           [r for r in grounding_df[1]],
                                           [w for w in grounding_df[3]],
                                           ))
                self.n_groundings = len(self.groundings)
                print('grounded triples size: {}'.format(self.n_groundings))
            print('------------------------------------')
            print('')

        idx = 0
        for i in self.training_triples:  # i[0], i[1], i[2] are all ids [h,t,r]
            self.train_triple_filter[i[0], i[1], i[2]] = 1
            self.train_val_triple_filter[i[0], i[1], i[2]] = 1
            self.train_val_test_triple_filter[i[0], i[1], i[2]] = 1
            self.all_pos_triples[i[0], i[1], i[2]] = 1
            self.triple2idx[i[0], i[1], i[2]] = idx
            self.all_pos_but_test_triples[i[0], i[1], i[2]] = 1
            idx = idx + 1
        for i in self.validation_triples:
            self.train_val_triple_filter[i[0], i[1], i[2]] = 1
            self.train_val_test_triple_filter[i[0], i[1], i[2]] = 1
            self.all_pos_triples[i[0], i[1], i[2]] = 1
            self.all_pos_but_test_triples[i[0], i[1], i[2]] = 1

        for i in self.test_triples:
            self.train_val_test_triple_filter[i[0], i[1], i[2]] = 1

        if self.train_with_psl:
            for i in self.psl_triples:
                self.all_pos_triples[i[0], i[1], i[2]] = 1
                self.all_pos_but_test_triples[i[0], i[1], i[2]] = 1

        if self.train_with_groundings:
            for i in self.groundings:
                self.all_pos_triples[i[0], i[1], i[2]] = 1
                self.all_pos_but_test_triples[i[0], i[1], i[2]] = 1

    #  for evaluation
    def load_hr_map(self, test_triples, training_triples, validation_triples):

        self.hr_map = {}
        # base query triples
        for triple in test_triples:
            h = triple[0]
            t = triple[1]
            r = triple[2]
            w = float(triple[3])
            # construct hr_map
            if self.hr_map.get(h) == None:
                self.hr_map[h] = {}
            if self.hr_map[h].get(r) == None:
                self.hr_map[h][r] = {t: w}
            else:
                self.hr_map[h][r][t] = w

        count = 0
        for h in self.hr_map:
            count += len(self.hr_map[h])
        print('Loaded ranking test queries. Number of (h,r,?t) queries: %d' % count)

        # additional data - training
        for triple in training_triples:
            h = triple[0]
            t = triple[1]
            r = triple[2]
            w = float(triple[3])

            # update hr_map
            if h in self.hr_map and r in self.hr_map[h]:
                self.hr_map[h][r][t] = w

        # additional data - validation
        for triple in validation_triples:
            h = triple[0]
            t = triple[1]
            r = triple[2]
            w = float(triple[3])

            # update hr_map
            if h in self.hr_map and r in self.hr_map[h]:
                self.hr_map[h][r][t] = w

        count = 0
        for h in self.hr_map:
            count += len(self.hr_map[h])
        print('Loadeded suplimentsss into self.hr_map %d' % count)

        return

    #  for evaluation
    def get_fixed_hr(self, n, outputdir=None):
        hr_map500 = {}
        dict_keys = []
        for h in self.hr_map.keys():
            for r in self.hr_map[h].keys():
                dict_keys.append([h, r])

        dict_keys = sorted(dict_keys, key=lambda x: len(self.hr_map[x[0]][x[1]]), reverse=True)
        dict_final_keys = []

        #for i in range(2525):# test data size
        for i in range(len(dict_keys)):
            dict_final_keys.append(dict_keys[i])

        count = 0
        for i in range(n):
            temp_key = random.choice(dict_final_keys)
            h = temp_key[0]
            r = temp_key[1]
            for t in self.hr_map[h][r]:
                w = self.hr_map[h][r][t]
                if hr_map500.get(h) == None:
                    hr_map500[h] = {}
                if hr_map500[h].get(r) == None:
                    hr_map500[h][r] = {t: w}
                else:
                    hr_map500[h][r][t] = w

        for h in hr_map500.keys():
            for r in hr_map500[h].keys():
                count = count + 1

        self.hr_map_sub = hr_map500

        # if outputdir is not None:
        #    with open(outputdir, 'wb') as handle:
        #        pickle.dump(hr_map500, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return hr_map500

    def next_raw_batch(self, batch_size):
        rand_idx = np.random.permutation(self.n_training_triple)
        start = 0
        while start < self.n_training_triple:  # Make sure the triples are not duplicating in training
            end = min(start + batch_size, self.n_training_triple)
            for i in rand_idx[start:end]:
                yield [self.training_triples[i]]
            start = end

    def generate_training_batch(self, in_queue, out_queue):
        while True:
            raw_batch = in_queue.get()
            if raw_batch is None:
                return
            else:
                batch_pos = raw_batch
                batch_neg = []
                corrupt_head_prob = np.random.binomial(1, 0.5)
                for head, tail, relation in batch_pos:
                    head_neg = head
                    tail_neg = tail
                    while True:
                        if corrupt_head_prob:
                            head_neg = random.choice(self.entities)
                        else:
                            tail_neg = random.choice(self.entities)
                        if (head_neg, tail_neg, relation) not in self.training_triple_pool:
                            break
                    batch_neg.append((head_neg, tail_neg, relation))
                out_queue.put((batch_pos, batch_neg))


# order is updated as [h,t,r,w] --


def gen_psl_samples(softlogics, n_psl_triples):
    # triples from probabilistic soft logic
    softlogics = np.asarray(softlogics, dtype=np.float64)
    # triple_indices = np.random.randint(0, len(softlogics), size=n_psl_triples)
    triple_indices = np.random.randint(0, softlogics.shape[0], size=n_psl_triples)
    samples = softlogics[triple_indices, :]
    return samples


def get_minibatches(X, mb_size, shuffle=True):
    """
    Generate minibatches from given dataset for training.

    Params:
    -------
    X: np.array of M x 3
        Contains the triplets from dataset. The entities and relations are
        translated to its unique indices.

    mb_size: int
        Size of each minibatch.

    shuffle: bool, default True
        Whether to shuffle the dataset before dividing it into minibatches.

    Returns:
    --------
    mb_iter: generator
        Example usage:
        --------------
        mb_iter = get_minibatches(X_train, mb_size)
        for X_mb in mb_iter:
            // do something with X_mb, the minibatch
    """
    X_shuff = X.copy()
    if shuffle:
        X_shuff = skshuffle(X_shuff)

    for i in range(0, X_shuff.shape[0], mb_size):
        yield X_shuff[i:i + mb_size]


def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
