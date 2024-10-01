from .earlytrain import EarlyTrain
import torch
import numpy as np
from .methods_utils import euclidean_dist, cossim
from ..nets.nets_utils import MyDataParallel
import os
import math
from utils import *
import deepcore.nets as nets
import math
import time

class RoP(EarlyTrain):
    def __init__(self, loader, configs, args, robust_learner, fraction=0.5, random_seed=None, epochs=0,
                 selection_method="LeastConfidence", balance: bool = False, metric="cossim",
                 torchvision_pretrain: bool = False, **kwargs):
        super().__init__(loader, configs, args, robust_learner, fraction, random_seed, epochs=epochs,
                         torchvision_pretrain=torchvision_pretrain, **kwargs)

        self.min_distances = None
        self.loader = loader
        self.metric_name = metric
        if metric == "euclidean":
            self.metric = euclidean_dist
        elif metric == "cossim":
            self.metric = cossim
        self.balance = balance
        self.selection_method = selection_method

    def num_classes_mismatch(self):
        raise ValueError("num_classes of pretrain dataset does not match that of the training dataset.")

    def while_update(self, outputs, loss, targets, epoch, batch_idx, batch_size):
        if batch_idx % self.args.print_freq == 0:
            print('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f' % (
            epoch, self.epochs, batch_idx + 1, (self.n_train // batch_size) + 1, loss.item()))

    def get_prob_embedding(self):
        print("Getting probs & embs!-------")
        self.model.eval()
        self.model.no_grad = True
        with torch.no_grad():
            with self.model.embedding_recorder:
                probs, embs = [], []
                # eval_train, eval_train_strong
                data_loader = self.loader.run('eval_train_strong')
                for i, data in enumerate(data_loader):
                    inputs = data[0]
                    output = self.model(inputs.to(self.args.device))
                    prob = torch.nn.functional.softmax(output.data, dim=1)

                    probs.append(prob.half())
                    embs.append(self.model.embedding_recorder.embedding.half())

                    if i%1000==0:
                        print("emb_batch: ", i)

        self.model.no_grad = False
        
        return torch.cat(probs, dim=0), torch.cat(embs, dim=0)

    def before_run(self):
        self.emb_dim = self.model.get_last_layer().in_features

    def finish_run(self):
        if isinstance(self.model, MyDataParallel):
            self.model = self.model.module
    
    def k_neighbor_confidence_greedy(self, probs, embs, budget: int, metric, device, random_seed=None, index=None,
                                      print_freq: int = 20):
        
        if type(embs) == torch.Tensor:
            assert embs.dim() == 2
        elif type(embs) == np.ndarray:
            assert embs.ndim == 2
            matrix = torch.from_numpy(embs).requires_grad_(False).to(device)

        sample_num = embs.shape[0]

        assert sample_num >= 1

        if budget < 0:
            raise ValueError("Illegal budget size.")

        assert callable(metric)


        if self.args.balance == True:
            print("balanced sampling!")
            available_classes = np.arange(self.args.n_class)

            noisy_labels = torch.tensor(self.loader.train_dataset.targets)
            
            num_sample_per_class, target_class_idxs_list = [], []

            each_class_budget = round((self.n_train * 0.8)/self.args.n_class)
            probs, _ = self.get_prob_embedding() 

            
            # obtain featuring embedding 
            trainFeatures = embs
            num_batch =  math.ceil(float(trainFeatures.size(0)) / (self.args.batch_size))
            sver_collection = []
            num_neighbor = self.args.num_neighbor
            num_class = self.args.n_class
            t1 = time.time()
            def get_adjmatrix(features, neighor_index):
                feature = features[neighor_index] 
                cosine_sim_matrix = torch.nn.functional.cosine_similarity(feature.unsqueeze(1), feature.unsqueeze(0), dim=2).to(self.args.device)
                matrix = 1 - cosine_sim_matrix
                D = matrix.sum(0) 
                D_norm = torch.diag(torch.pow(D, -0.5)).to(self.args.device)
                E = torch.matmul(D_norm, torch.matmul(matrix, D_norm))
                E = torch.eye(E.shape[0]).to(self.args.device) + E
                return E
            from scipy.stats import mode
            for batch_idx in range(num_batch):

                # obtain KNN
                features = trainFeatures[batch_idx * self.args.batch_size : min((batch_idx+1) * self.args.batch_size, trainFeatures.size(0))]
                batch_noisy_labels = noisy_labels[batch_idx * self.args.batch_size : min((batch_idx+1) * self.args.batch_size, trainFeatures.size(0))].view(-1)
                dist = torch.mm(features, trainFeatures.t())
                dist[torch.arange(dist.size(0)), torch.arange(dist.size(0))] = -1
                _, batch_neighbors_index = dist.topk(num_neighbor, dim=1, largest=True, sorted=True)
                neighbors = batch_neighbors_index.view(-1)

                # feature propagation 
                if (batch_idx+1) * self.args.batch_size < trainFeatures.size(0):
                    weight_adj = torch.zeros((self.args.batch_size, num_neighbor, num_neighbor)).to(self.args.device)
                    neighor_logits = torch.zeros((self.args.batch_size, num_neighbor, self.args.n_class)).to(self.args.device)
                else:
                    surplus =  trainFeatures.size(0) - batch_idx * self.args.batch_size
                    weight_adj = torch.zeros((surplus, num_neighbor, num_neighbor)).to(self.args.device)
                    neighor_logits = torch.zeros((surplus, num_neighbor, self.args.n_class)).to(self.args.device)
                
                # label propagation
                for j in range(weight_adj.size(0)):
                    batch_neighbors_index[j].cpu().numpy()
                    weight_adj[j] = get_adjmatrix(trainFeatures, batch_neighbors_index[j])
                    neighor_logits[j] = weight_adj[j].float()@probs[batch_neighbors_index[j]].float()
                weight_adj = weight_adj.view(weight_adj.shape[0]*weight_adj.shape[1], weight_adj.shape[2])
    
                # obatin NLI-Score
                neigh_probs = F.softmax(neighor_logits, dim=-1)
                M, _ = features.shape
                given_labels = torch.full(size=(M, num_class), fill_value=0.0001).cuda()
                given_labels.scatter_(dim=1, index=torch.unsqueeze(batch_noisy_labels.long().to('cuda'), dim=1), value=1 - 0.0001)
                given_labels = given_labels.repeat(1, num_neighbor).view(-1, num_class)
                sver = js_div(neigh_probs.view(-1, num_class), given_labels)
                sver_collection += sver.view(-1, num_neighbor).mean(dim=1).cpu().numpy().tolist()
            sver_collection = np.array(sver_collection)
        t2 = time.time()


        # sampling 
        result_sver = []
        sver_collection = torch.tensor(sver_collection).to('cuda')
        each_class_budget1 = int(self.coreset_size/self.args.n_class)
        result_consistency = []
        import random
        for i in range(self.args.n_class):
            print("-- the number of this class is {}".format(len(true_indices_per_class))) 
            
            true_indices_per_class = target_class_idxs_list[i].to('cuda')
            value, index_i = torch.sort(sver_collection[true_indices_per_class], descending=False)
            
            n = len(index_i)
            k = each_class_budget1
            index = random.sample(range(n), min(n,k))
            save_index = true_indices_per_class[index_i][: each_class_budget1]
            save_index1 = true_indices_per_class[index_i][index]
            result_sver.append(save_index)
            result_consistency.append(sver_collection[save_index1])

        result_sver = torch.cat(result_sver).to('cuda')
        
        # divided into other classes
        mask = ~torch.isin(torch.arange(sver_collection.size(0)).to('cuda'), result_sver)
        residual_sever_collectioin = sver_collection[mask]
        _, other_index_all = torch.sort(residual_sever_collectioin, descending=False)
        other_num = self.coreset_size-result_sver.size(0)

        while other_num > 0:
            print("-- the number of remaining samples: ", other_num)
            for i in range(self.args.n_class):
                true_indices_per_class = target_class_idxs_list[i].to('cuda')
                save_class_mask = torch.isin(true_indices_per_class, result_sver)
                if len(true_indices_per_class)==save_class_mask.sum():
                    pass
                else:
                    residual_mask_class = ~torch.isin(true_indices_per_class, true_indices_per_class[save_class_mask])
                    residual_true_indices_per_class = true_indices_per_class[residual_mask_class]
                    _, residual_index_all = torch.sort(sver_collection[residual_true_indices_per_class], descending=False)

                    result_sver = torch.cat((result_sver, residual_index_all[0].unsqueeze(0)))
                    if other_num > 0:
                        other_num -= 1
                    else:
                        break

        for i in range(self.args.n_class):
            true_indices_per_class = target_class_idxs_list[i].to('cuda')
            save_class_mask = torch.isin(true_indices_per_class, result_sver)
            print("the class {} and the number of this class is {}".format(i, len(save_class_mask)))
        print('-- all selected samples: {}'.format(result_sver.size(0)), '-----residual samples: ', other_num)

        
        result_consistency = torch.cat(result_consistency).to('cuda')

        save_all_index = []
        with torch.no_grad():
            np.random.seed(random_seed)
            select_result = np.zeros(sample_num, dtype=bool)
       
        return sver_collection, np.arange(self.args.n_train)[np.array(result_sver.cpu())]
        
    def select(self, **kwargs):
        if self.robust_learner == 'SOP':
            _, configs =self.run()
        else:
            _, configs =self.run()
        # save
        if self.args.dataset in ['WebVision', 'Clothing1M', 'ImageNet']:
            probs_path = './RobustCoreLogs/' + str(self.args.dataset) + '/' + str(self.args.robust_learner) \
                         + '/probs_nr'+ str(self.args.noise_rate)+'.pth'
            embs_path = './RobustCoreLogs/' + str(self.args.dataset) + '/' + str(self.args.robust_learner) \
                        + '/embs'+ str(self.args.noise_rate)+'.pth'
            if os.path.isfile(probs_path) and os.path.isfile(embs_path):
                probs1 = torch.load(probs_path)
                embs = torch.load(embs_path)
                print("probs & embs loaded!")
            else:
                probs1, embs = self.get_prob_embedding()

                torch.save(probs1, probs_path)
                torch.save(embs, embs_path)
                print("probs & embs saved!")
        else:
            probs1, embs = self.get_prob_embedding()  # 50000,512
            print("----  embedding ----", embs.shape)
        sver_collection, selection_result1 = self.k_neighbor_confidence_greedy(probs1, embs, budget=self.coreset_size,
                                           metric=self.metric, device=self.args.device,
                                           random_seed=self.random_seed, print_freq=self.args.print_freq)
        probs, embs = None, None

        del self.model
        torch.cuda.empty_cache()

        return {"indices": selection_result1}, self.configs