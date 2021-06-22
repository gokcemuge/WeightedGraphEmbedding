import torch
import numpy as np
import torch.nn as nn
from torch.nn.init import xavier_normal_, xavier_uniform_
from torch.nn import functional as F
from torch.autograd import Variable
from utilities.data_utilities import truncated_normal_


class Model(nn.Module):
    def __init__(self, name, kg, embedding_dim, batch_size, learning_rate, neg_size, gpu=True, regul=False, train_with_groundings=False, train_with_psl=False, load_model=False, save_model=False):
        super(Model, self).__init__()

        self.name = name
        self.gpu = gpu
        self.kg = kg
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.neg_size = neg_size
        self.regul = regul
        self.train_with_groundings = train_with_groundings
        self.train_with_psl = train_with_psl
        self.save_model = save_model
        self.load_model = load_model

        # To implement a model score function should be defined and initialised in this class
        self.init_embedding()

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

        if self.gpu:
            self.cuda()

    def calculate_score(self, head, tail, relation):
        if self.gpu:
            h_i = Variable(torch.from_numpy(head).cuda())
            t_i = Variable(torch.from_numpy(tail).cuda())
            r_i = Variable(torch.from_numpy(relation).cuda())
        else:
            h_i = Variable(torch.from_numpy(head))
            t_i = Variable(torch.from_numpy(tail))
            r_i = Variable(torch.from_numpy(relation))

        if self.name == 'RotatE':

            pi = 3.14159265358979323846
            h_real = self.emb_E_real(h_i).view(-1, self.embedding_dim)
            t_real = self.emb_E_real(t_i).view(-1, self.embedding_dim)
            r_real = torch.cos(self.emb_R_phase(r_i).view(-1, self.embedding_dim)/(6 / np.sqrt(self.embedding_dim)/pi))

            h_img = self.emb_E_img(h_i).view(-1, self.embedding_dim)
            t_img = self.emb_E_img(t_i).view(-1, self.embedding_dim)
            r_img = torch.sin(self.emb_R_phase(r_i).view(-1, self.embedding_dim)/(6 / np.sqrt(self.embedding_dim)/pi))

            if self.L == 'L1':
                out_real = torch.sum(torch.abs(h_real*r_real-h_img*r_img-t_real),1)
                out_img  = torch.sum(torch.abs(h_real*r_img+h_img*r_real-t_img),1)
                out = out_real + out_img
            else:
                out_real = torch.sum((h_real*r_real-h_img*r_img-t_real) ** 2, 1)
                out_img = torch.sum((h_real*r_img+h_img*r_real-t_img) ** 2, 1)
                out = torch.sqrt(out_img + out_real)

        elif self.name == 'distmult':

            h = self.emb_E(h_i).view(-1, self.embedding_dim)
            t = self.emb_E(t_i).view(-1, self.embedding_dim)
            r = self.emb_R(r_i).view(-1, self.embedding_dim)

            out = torch.sum(h * t * r, dim=1)

            if self.regul:
                self.regul = torch.mean(h ** 2) + torch.mean(t ** 2) + torch.mean(r ** 2)
                return out, self.regul

            return out

        elif self.name == 'complEx':
            head = self.emb_E(h_i).view(-1, self.embedding_dim)
            tail = self.emb_E(t_i).view(-1, self.embedding_dim)
            relation = self.emb_R(r_i).view(-1, self.embedding_dim)
            re_head, im_head = torch.chunk(head, 2, dim=1)
            re_relation, im_relation = torch.chunk(relation, 2, dim=1)
            re_tail, im_tail = torch.chunk(tail, 2, dim=1)
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

            out = torch.sum(score, dim=1)
            if self.regul:
                self.regul = torch.mean(re_head ** 2) + torch.mean(im_head ** 2) + torch.mean(re_tail ** 2) + torch.mean(
                    im_tail ** 2) + torch.mean(re_relation ** 2) + torch.mean(im_relation ** 2)
                return out, self.regul

            return out

        elif self.name == 'transE':
            h = self.emb_E(h_i).view(-1, self.embedding_dim)
            t = self.emb_E(t_i).view(-1, self.embedding_dim)
            r = self.emb_R(r_i).view(-1, self.embedding_dim)
            out = torch.sqrt(torch.sum((h + r - t) ** 2, 1))
            return out

        elif self.name == 'rescal':

            h = self.emb_E(h_i).view(-1, self.embedding_dim)
            t = self.emb_E(t_i).view(-1, self.embedding_dim, 1)
            r = self.emb_R(r_i).view(-1, self.embedding_dim, self.embedding_dim)

            tr = torch.matmul(r, t)
            tr = tr.view(-1, self.embedding_dim)
            out = torch.sum(h * tr, dim=1)

            if self.regul:
                regul = (torch.mean(h ** 2) + torch.mean(t ** 2) + torch.mean(r ** 2)) / 3
                return out, self.regul
            return out

        elif self.name == 'UKGE_logi':
            h = self.emb_E(h_i).view(-1, self.embedding_dim)
            t = self.emb_E(t_i).view(-1, self.embedding_dim)
            r = self.emb_R(r_i).view(-1, self.embedding_dim)

            plausibility = torch.sum(r * (h * t), dim=1)
            out = self.sigmoid(self.weight*plausibility+self.bias) #logistic regression

        elif self.name == 'UKGE_rect':
            h = self.emb_E(h_i).view(-1, self.embedding_dim)
            t = self.emb_E(t_i).view(-1, self.embedding_dim)
            r = self.emb_R(r_i).view(-1, self.embedding_dim)

            plausibility = torch.sum(r * (h * t), dim=1)
            out = (self.weight*plausibility+self.bias) # no CLAMP, they will be later
            # original implementation uses l2_loss for regularizer
            # In UKGE, it is said that regularizer is not being used during the testing phase
            if self.regul:
                regularizer = torch.mean(h ** 2) / 2 + torch.mean(t ** 2) / 2 + torch.mean(r ** 2) / 2
                return out, regularizer

        elif self.name == 'WGE_logi':
            h = self.emb_E(h_i).view(-1, self.embedding_dim)
            t = self.emb_E(t_i).view(-1, self.embedding_dim)
            r = self.emb_R(r_i).view(-1, self.embedding_dim)

            plausibility = torch.sum(r * (h * t), dim=1)  # DistMult
            out = self.sigmoid(plausibility)
            out = out.view(1, -1)

        elif self.name == 'WGE_rect':
            h = self.emb_E(h_i).view(-1, self.embedding_dim)  # h is embedding vector
            t = self.emb_E(t_i).view(-1, self.embedding_dim)  # h_i index of head entity
            r = self.emb_R(r_i).view(-1, self.embedding_dim)

            plausibility = torch.sum(r * (h * t), dim=1)
            out = plausibility.view(1,-1)

            if self.regul:
                regularizer = torch.mean(h ** 2) / 2 + torch.mean(t ** 2) / 2 + torch.mean(r ** 2) / 2
                return out, regularizer

        elif self.name == 'WGE_logi_rescal':
            h = self.emb_E(h_i).view(-1, self.embedding_dim)
            t = self.emb_E(t_i).view(-1, self.embedding_dim, 1)
            r = self.emb_R(r_i).view(-1, self.embedding_dim, self.embedding_dim)

            tr = torch.matmul(r, t)
            tr = tr.view(-1, self.embedding_dim)
            out = torch.sum(h * tr, dim=1)
            out = self.sigmoid(out)
            out = out.view(1, -1)

        elif self.name == 'WGE_rect_rescal':
            h = self.emb_E(h_i).view(-1, self.embedding_dim)
            t = self.emb_E(t_i).view(-1, self.embedding_dim, 1)
            r = self.emb_R(r_i).view(-1, self.embedding_dim, self.embedding_dim)

            tr = torch.matmul(r, t)
            tr = tr.view(-1, self.embedding_dim)
            out = torch.sum(h * tr, dim=1)

            if self.regul:
                regularizer = torch.mean(h ** 2) / 2 + torch.mean(t ** 2) / 2
                return out, regularizer

        return out

    def init_embedding(self):

        if self.name == 'RotatE':
            self.emb_E_real = torch.nn.Embedding(self.kg.n_entity, self.embedding_dim, padding_idx=0)
            self.emb_E_img = torch.nn.Embedding(self.kg.n_entity, self.embedding_dim, padding_idx=0)
            self.emb_R_phase = torch.nn.Embedding(self.kg.n_relation, self.embedding_dim, padding_idx=0)
            self.error = torch.nn.Embedding(self.kg.n_training_triple, 1)
            r = 6 / np.sqrt(self.embedding_dim)
            self.emb_E_real.weight.data.uniform_(-r, r)
            self.emb_E_img.weight.data.uniform_(-r, r)
            self.emb_R_phase.weight.data.uniform_(-r, r)
            self.error.weight.data.uniform_(0, r)

        elif self.name == 'distmult':
            self.emb_E = torch.nn.Embedding(self.kg.n_entity, self.embedding_dim, padding_idx=0)
            self.emb_R = torch.nn.Embedding(self.kg.n_relation, self.embedding_dim, padding_idx=0)
            xavier_normal_(self.emb_E.weight.data)
            xavier_normal_(self.emb_R.weight.data)

        elif self.name == 'complEx':
            self.emb_E = nn.Embedding(self.kg.n_entity, self.embedding_dim)
            self.emb_R = nn.Embedding(self.kg.n_relation, self.embedding_dim)
            self.embeddings = [self.emb_E, self.emb_R]
            xavier_normal_(self.emb_E.weight.data)
            xavier_normal_(self.emb_R.weight.data)

        elif self.name == 'rescal':
            self.emb_E = nn.Embedding(self.kg.n_entity, self.embedding_dim)
            self.emb_R = nn.Embedding(self.kg.n_relation, self.embedding_dim*self.embedding_dim)
            xavier_uniform_(self.emb_E.weight.data)
            xavier_uniform_(self.emb_R.weight.data)

        elif self.name == 'transE':
            self.emb_E = torch.nn.Embedding(self.kg.n_entity, self.embedding_dim, padding_idx=0)
            self.emb_R = torch.nn.Embedding(self.kg.n_relation, self.embedding_dim, padding_idx=0)
            # Initialization
            r = 6 / np.sqrt(self.embedding_dim)
            self.emb_E.weight.data.uniform_(-r, r)
            self.emb_R.weight.data.uniform_(-r, r)
            self.normalize_embeddings()

        elif self.name == 'UKGE_logi':
            self.emb_E = torch.nn.Embedding(self.kg.n_entity, self.embedding_dim, padding_idx=0).double()
            self.emb_R = torch.nn.Embedding(self.kg.n_relation, self.embedding_dim, padding_idx=0).double()
            #xavier_normal_(self.emb_E.weight.data)
            #xavier_normal_(self.emb_R.weight.data)

            truncated_normal_(self.emb_E.weight.data, mean=0, std=0.3)
            truncated_normal_(self.emb_R.weight.data, mean=0, std=0.3)

            #self.weight = torch.nn.Parameter(torch.from_numpy(np.random.rand(1).astype(np.float64)).cuda())
            self.weight = torch.nn.Parameter(torch.zeros([1, 1], dtype=torch.float64).cuda())
            self.weight.requires_grad = True
            #self.bias = torch.nn.Parameter(torch.from_numpy(np.random.rand(1).astype(np.float64)).cuda())
            self.bias = torch.nn.Parameter(torch.zeros([1, 1], dtype=torch.float64).cuda())
            self.bias.requires_grad = True

        elif self.name == 'UKGE_rect':
            self.emb_E = torch.nn.Embedding(self.kg.n_entity, self.embedding_dim, padding_idx=0).double()
            self.emb_R = torch.nn.Embedding(self.kg.n_relation, self.embedding_dim, padding_idx=0).double()
            #xavier_normal_(self.emb_E.weight.data)
            #xavier_normal_(self.emb_R.weight.data)
            truncated_normal_(self.emb_E.weight.data, mean=0, std=0.3)
            truncated_normal_(self.emb_R.weight.data, mean=0, std=0.3)
            #self.weight = torch.nn.Parameter(torch.from_numpy(np.random.rand(1).astype(np.float64)).cuda())
            self.weight = torch.nn.Parameter(torch.zeros([1, 1], dtype=torch.float64).cuda())
            self.weight.requires_grad = True
            #self.bias = torch.nn.Parameter(torch.from_numpy(np.random.rand(1).astype(np.float64)).cuda())
            self.bias = torch.nn.Parameter(torch.zeros([1, 1], dtype=torch.float64).cuda())
            self.bias.requires_grad = True

        elif self.name == 'WGE_rect' or self.name == 'WGE_logi':
            self.emb_E = torch.nn.Embedding(self.kg.n_entity, self.embedding_dim, padding_idx=0).double()
            self.emb_R = torch.nn.Embedding(self.kg.n_relation, self.embedding_dim, padding_idx=0).double()

            # truncated_normal_(self.emb_E.weight.data, mean=0, std=0.3)
            # truncated_normal_(self.emb_R.weight.data, mean=0, std=0.3)

            xavier_normal_(self.emb_E.weight.data)
            xavier_normal_(self.emb_R.weight.data)

            self.epsilons_left = torch.nn.Embedding(self.kg.n_training_triple, (self.neg_size + 1),
                                                    padding_idx=0).double()
            self.epsilons_right = torch.nn.Embedding(self.kg.n_training_triple, (self.neg_size + 1),
                                                     padding_idx=0).double()

            xavier_normal_(self.epsilons_left.weight.data)
            xavier_normal_(self.epsilons_right.weight.data)

        elif self.name == 'WGE_rect_rescal' or self.name == 'WGE_logi_rescal':

            # for rescal part
            self.emb_E = nn.Embedding(self.kg.n_entity, self.embedding_dim)
            self.emb_R = nn.Embedding(self.kg.n_relation, self.embedding_dim * self.embedding_dim)
            xavier_uniform_(self.emb_E.weight.data)
            xavier_uniform_(self.emb_R.weight.data)

            self.epsilons_left = torch.nn.Embedding(self.kg.n_training_triple, (self.neg_size + 1),
                                                    padding_idx=0).double()
            self.epsilons_right = torch.nn.Embedding(self.kg.n_training_triple, (self.neg_size + 1),
                                                     padding_idx=0).double()

            xavier_normal_(self.epsilons_left.weight.data)
            xavier_normal_(self.epsilons_right.weight.data)

    def forward(self, x):
        head, tail, relation = x[:, 0].astype(np.int64), x[:, 1].astype(np.int64), x[:, 2].astype(np.int64)
        if self.regul:
            out, regularizer = self.calculate_score(head, tail, relation)
            return out, regularizer

        out = self.calculate_score(head, tail, relation)
        return out

    def normalize_embeddings(self):
        self.emb_E.weight.data.renorm_(p=2, dim=0, maxnorm=1)
        self.emb_R.weight.data.renorm_(p=2, dim=0, maxnorm=1)

    def get_embeddings(self):
        entity_embedding = self.embeddings[0].weight.data.numpy()
        relation_embedding = self.embeddings[1].weight.data.numpy()
        return entity_embedding, relation_embedding

    def get_embedding(self, index):
        index = index.astype(np.int64)
        embedding = self.emb_E(torch.tensor(index).cuda())
        return embedding

    def get_vectorized_embedding(self, vector):
        if self.gpu:
            vector_emb = Variable(torch.from_numpy(vector).cuda())
        else:
            vector_emb = Variable(torch.from_numpy(vector))
        embedding_vector = self.emb_E(vector_emb).view(-1, self.embedding_dim)
        return embedding_vector

    def load_embedding(self):
        path = 'saved_models/' + str(self.name) + '/'

        if self.train_with_groundings:
            path = 'saved_models/' + str(self.name) + '_with_rule' + '/'

        if self.name == 'UKGE_logi' or self.name == 'UKGE_rect':
            self.emb_E.weight.data = torch.from_numpy(np.load(str(path) + str(self.name) + '_emb_E'+ '.npy')).cuda()
            self.emb_R.weight.data = torch.from_numpy(np.load(str(path) + str(self.name) + '_emb_R'+ '.npy')).cuda()
            self.weight.data = torch.from_numpy(np.load(str(path) + str(self.name) + '_weight'+ '.npy')).cuda()
            self.bias.data = torch.from_numpy(np.load(str(path) + str(self.name) + '_bias'+ '.npy')).cuda()

        if self.name == 'WGE_logi' or self.name == 'WGE_rect':
            self.emb_E.weight.data = torch.from_numpy(np.load(str(path) + str(self.name) + '_emb_E'+ '.npy')).cuda()
            self.emb_R.weight.data = torch.from_numpy(np.load(str(path) + str(self.name) + '_emb_R'+ '.npy')).cuda()
            self.epsilons_left.weight.data = torch.from_numpy(np.load(str(path) + str(self.name) + '_epsilons_left'+ '.npy')).cuda()
            self.epsilons_right.weight.data = torch.from_numpy(np.load(str(path) + str(self.name) + '_epsilons_right'+ '.npy')).cuda()

        if self.name == "transE" or self.name == 'distmult' or self.name == 'complEx' or self.name == 'rescal':
            self.emb_E.weight.data = torch.from_numpy(np.load(str(path) + str(self.name) + '_emb_E' + '.npy')).cuda()
            self.emb_R.weight.data = torch.from_numpy(np.load(str(path) + str(self.name) + '_emb_R' + '.npy')).cuda()

            if self.name == "complEx":
                self.embeddings = [self.emb_E, self.emb_R]

        print('model is loaded!')

    def save_embedding(self, epoch):
        path = 'saved_models/' + str(self.name) +'/'
        if self.train_with_groundings:
            path = 'saved_models/' + str(self.name) + '_with_rule' + '/'

        if self.name == 'UKGE_logi' or self.name == 'UKGE_rect':
            emb_E = self.emb_E.weight.detach().cpu().numpy()
            emb_R = self.emb_R.weight.detach().cpu().numpy()
            weight = self.weight.detach().cpu().numpy()
            bias = self.bias.detach().cpu().numpy()

            np.save(str(path) + str(self.name) + '_emb_E' + str(epoch), emb_E )
            np.save(str(path) + str(self.name) + '_emb_R' + str(epoch), emb_R)
            np.save(str(path) + str(self.name) + '_weight' + str(epoch), weight)
            np.save(str(path) + str(self.name) + '_bias' + str(epoch), bias)

        if self.name == 'WGE_logi' or self.name == 'WGE_rect':
            emb_E = self.emb_E.weight.detach().cpu().numpy()
            emb_R = self.emb_R.weight.detach().cpu().numpy()
            epsilons_left = self.epsilons_left.weight.detach().cpu().numpy()
            epsilons_right = self.epsilons_right.weight.detach().cpu().numpy()

            np.save(str(path) + str(self.name) + '_emb_E_' + str(epoch), emb_E )
            np.save(str(path) + str(self.name) + '_emb_R_' + str(epoch), emb_R)
            np.save(str(path) + str(self.name) + '_epsilons_left_' + str(epoch), epsilons_left)
            np.save(str(path) + str(self.name) + '_epsilons_right_' + str(epoch), epsilons_right)

        if self.name == "transE" or self.name == 'distmult' or self.name == 'complEx' or self.name == 'rescal':
            emb_E = self.emb_E.weight.detach().cpu().numpy()
            emb_R = self.emb_R.weight.detach().cpu().numpy()

            np.save(str(path) + str(self.name) + '_emb_E_' + str(epoch), emb_E)
            np.save(str(path) + str(self.name) + '_emb_R_' + str(epoch), emb_R)

        print('model is saved!')
