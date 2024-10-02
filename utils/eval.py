import torch
import torch.nn.functional as F
from torchmetrics.metric import Metric
from einops import rearrange
import numpy as np
from tqdm import tqdm
from snapml import LogisticRegression
from sklearn.metrics import accuracy_score
from typing import Tuple
import gc

class WeightedKNNClassifier(Metric):
    def __init__(
        self,
        k: int = 20,
        T: float = 0.07,
        max_distance_matrix_size: int = int(5e6),
        distance_fx: str = "cosine",
        epsilon: float = 0.00001,
        dist_sync_on_step: bool = False,
    ):
        """Implements the weighted k-NN classifier used for evaluation.

        Args:
            k (int, optional): number of neighbors. Defaults to 20.
            T (float, optional): temperature for the exponential. Only used with cosine
                distance. Defaults to 0.07.
            max_distance_matrix_size (int, optional): maximum number of elements in the
                distance matrix. Defaults to 5e6.
            distance_fx (str, optional): Distance function. Accepted arguments: "cosine" or
                "euclidean". Defaults to "cosine".
            epsilon (float, optional): Small value for numerical stability. Only used with
                euclidean distance. Defaults to 0.00001.
            dist_sync_on_step (bool, optional): whether to sync distributed values at every
                step. Defaults to False.
        """

        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.k = k
        self.T = T
        self.max_distance_matrix_size = max_distance_matrix_size
        self.distance_fx = distance_fx
        self.epsilon = epsilon

        self.add_state("train_features", default=[], persistent=False)
        self.add_state("train_targets", default=[], persistent=False)
        self.add_state("test_features", default=[], persistent=False)
        self.add_state("test_targets", default=[], persistent=False)

    def update(
        self,
        train_features: torch.Tensor = None,
        train_targets: torch.Tensor = None,
        test_features: torch.Tensor = None,
        test_targets: torch.Tensor = None,
    ):
        """Updates the memory banks. If train (test) features are passed as input, the
        corresponding train (test) targets must be passed as well.

        Args:
            train_features (torch.Tensor, optional): a batch of train features. Defaults to None.
            train_targets (torch.Tensor, optional): a batch of train targets. Defaults to None.
            test_features (torch.Tensor, optional): a batch of test features. Defaults to None.
            test_targets (torch.Tensor, optional): a batch of test targets. Defaults to None.
        """
        assert (train_features is None) == (train_targets is None)
        assert (test_features is None) == (test_targets is None)

        if train_features is not None:
            assert train_features.size(0) == train_targets.size(0)
            self.train_features.append(train_features.detach())
            self.train_targets.append(train_targets.detach())

        if test_features is not None:
            assert test_features.size(0) == test_targets.size(0)
            self.test_features.append(test_features.detach())
            self.test_targets.append(test_targets.detach())

    @torch.no_grad()
    def compute(self) -> Tuple[float]:
        """Computes weighted k-NN accuracy @1 and @5. If cosine distance is selected,
        the weight is computed using the exponential of the temperature scaled cosine
        distance of the samples. If euclidean distance is selected, the weight corresponds
        to the inverse of the euclidean distance.

        Returns:
            Tuple[float]: k-NN accuracy @1 and @5.
        """

        train_features = torch.cat(self.train_features)
#         print(train_features.shape)
        train_targets = torch.cat(self.train_targets)
        test_features = torch.cat(self.test_features)
        test_targets = torch.cat(self.test_targets)

        if self.distance_fx == "cosine":
            train_features = F.normalize(train_features)
            test_features = F.normalize(test_features)

        num_classes = torch.unique(test_targets).numel()
        num_train_images = train_targets.size(0)
        num_test_images = test_targets.size(0)
        num_train_images = train_targets.size(0)
        chunk_size = min(
            max(1, self.max_distance_matrix_size // num_train_images),
            num_test_images,
        )
        k = min(self.k, num_train_images)

        top1, top5, total = 0.0, 0.0, 0
        retrieval_one_hot = torch.zeros(k, num_classes).to(train_features.device)
        
        for idx in range(0, num_test_images, chunk_size):
            # get the features for test images
            features = test_features[idx : min((idx + chunk_size), num_test_images), :]
            targets = test_targets[idx : min((idx + chunk_size), num_test_images)]
            batch_size = targets.size(0)

            # calculate the dot product and compute top-k neighbors
            if self.distance_fx == "cosine":
                similarities = torch.mm(features, train_features.t())
            elif self.distance_fx == "euclidean":
                similarities = 1 / (torch.cdist(features, train_features) + self.epsilon)
            else:
                raise NotImplementedError

            similarities, indices = similarities.topk(k, largest=True, sorted=True)
            candidates = train_targets.view(1, -1).expand(batch_size, -1)
            retrieved_neighbors = torch.gather(candidates, 1, indices)

            retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
            retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)

            if self.distance_fx == "cosine":
                similarities = similarities.clone().div_(self.T).exp_()

            probs = torch.sum(
                torch.mul(
                    retrieval_one_hot.view(batch_size, -1, num_classes),
                    similarities.view(batch_size, -1, 1),
                ),
                1,
            )
            _, predictions = probs.sort(1, True)
            # find the predictions that match the target
            correct = predictions.eq(targets.data.view(-1, 1))
            top1 = top1 + correct.narrow(1, 0, 1).sum().item()
            top5 = (
                top5 + correct.narrow(1, 0, min(5, k, correct.size(-1))).sum().item()
            )  # top5 does not make sense if k < 5
            total += targets.size(0)

        top1 = top1 * 100.0 / total
        top5 = top5 * 100.0 / total

        self.reset()

        return top1, top5
    
def knn_classify(args, train_features, train_labels, test_features, test_labels):
    """Perform k-Nearest Neighbor classification using cosine similaristy as metric.

    Options:
        k (int): top k features for kNN
    
    """
    train_features = F.normalize(train_features, dim=1)
    test_features = F.normalize(test_features, dim=1)
    sim_mat = train_features @ test_features.T
    topk = sim_mat.topk(k=args.k, dim=0)
    topk_pred = train_labels[topk.indices]
    test_pred = topk_pred.mode(0).values.detach()
    acc = compute_accuracy(test_pred.numpy(), test_labels.numpy())
    print("kNN: {}".format(acc))
    return acc

def compute_accuracy(y_pred, y_true):
    """Compute accuracy by counting correct classification. """
    assert y_pred.shape == y_true.shape
    return 1 - np.count_nonzero(y_pred - y_true) / y_true.size


def eval_model(model,load_batch_size=400):
    _ = model.eval()
    #     m = torch.nn.AvgPool2d(3,1)
    m = torch.nn.AvgPool2d(1)
    fea_ls = []
    i_ls = []

    for _ in range(1):
        for x,i in train_dl:
#             x=gperm(x)
            x = x.to(device)
            x = sample(x,kernel_pyramid,mask)
            features = model(x).cpu().detach()[:,1:,:]
#             features = rearrange(features,'b (h w) c -> b c h w ',h=8)
            features = features.reshape(load_batch_size,-1)
#             features = features
            fea_ls.append(features)
            i_ls.append(i)
    X = torch.cat(fea_ls).numpy()
    y = torch.cat(i_ls).numpy()
    fea_ls_test = []
    i_ls_test = []
    for x,i in test_dl:
#         if args.permute:
#         x=gperm(x)
        x = x.to(device)
        x =sample(x,kernel_pyramid,mask)
        features = model(x).cpu().detach()[:,1:,:]
#         features = rearrange(features,'b (h w) c -> b c h w ',h=8)
        features = features.reshape(load_batch_size,-1)
#         features = features
        fea_ls_test.append(features)
        i_ls_test.append(i)
    X_test = torch.cat(fea_ls_test).numpy()
    y_test = torch.cat(i_ls_test).numpy()
    clf = LogisticRegression(use_gpu = True, device_ids = list(range(n_gpus)), fit_intercept=True,penalty='l2',batch_size=1024,regularizer=1)
    clf.fit(X,y)
    y_test_hat = clf.predict(X_test)
    acc_score = accuracy_score(y_test,y_test_hat)
    _ = model.train()
    return acc_score

# def eval_model_KNN(model,load_batch_size=500):
    
#     _=model.eval()
#     BASIS1_NUM=args.emb_dim
#     V = torch.zeros(BASIS1_NUM,BASIS1_NUM).to(device)
#     BATCH_NUM = 0
#     # for e in range(epochs):
#     for x,i in train_dl:
#         x=gperm(x)
#         x = x.to(device)
#         features = model(x).detach()[:,1:,:]
#         b,l,c=features.shape
#         ahat = rearrange(features,"b l c -> c (b l)")
#         V = V + torch.mm(ahat,ahat.t())/(b*l)
#         BATCH_NUM +=1
#     V = V/BATCH_NUM

#     w1, v1 = torch.linalg.eigh(V + torch.eye(BASIS1_NUM).to(device) * 1e-7, UPLO='U')
#     whiteMat = torch.mm((w1.add(1e-3).pow(-0.5)).diag(), v1.t())
#     colorMat = torch.mm(v1, w1.add(1e-3).pow(0.5).diag())

#     X = torch.zeros(50000,l,c)
#     X_test = torch.zeros(10000,l,c)
    
#     idx = 0
#     for x,i in train_dl:
#         x=gperm(x)
#         x = x.to(device)
#         features = model(x).detach()[:,1:,:]
#         b,l,c = features.shape
#         features_batch = rearrange(features,"b l c ->c (b l)")
#         features_batch = torch.mm(whiteMat,features_batch)
#         features = rearrange(features_batch,"c (b l) ->b l c", b = b)
#         features = features.div(features.norm(dim=1,keepdim=True))
#         X[idx:idx+load_batch_size,...] = features.cpu()
#         idx +=load_batch_size
#     idx = 0
#     for x,i in test_dl:
#         x=gperm(x)
#         x = x.to(device)
#         features = model(x).detach()[:,1:,:]
#         b,l,c = features.shape
#         features_batch = rearrange(features,"b l c ->c (b l)")
#         features_batch = torch.mm(whiteMat,features_batch)
#         features = rearrange(features_batch,"c (b l) ->b l c", b = b)
#         features = features.div(features.norm(dim=1,keepdim=True))
#         X_test[idx:idx+load_batch_size,...] = features.cpu()
#         idx +=load_batch_size
        
#     train_label = torch.tensor(train_dataset_eval.targets)
#     test_label = torch.tensor(val_dataset_eval.targets)
    
#     knn_c = WeightedKNNClassifier(k=30,T=0.03)
#     knn_c.update(X.flatten(1),train_label,X_test.flatten(1),test_label)
#     acc2 = knn_c.compute()
#     _ = model.train()
#     del X_test,X
#     gc.collect()
    
#     return acc2[0]


def eval_model_KNN(model,data_train,data_test,train_label,test_label, load_batch_size=400,device = "cuda:0"):
    
    _=model.eval()
    BASIS1_NUM=192
    V = torch.zeros(BASIS1_NUM,BASIS1_NUM).to(device)
    BATCH_NUM = 0
    # for e in range(epochs):
    for idx in tqdm(range(0,len(data_train),load_batch_size)):
        x = data_train[idx:idx+load_batch_size].to(device)
        features = model(x).detach()[:,1:,:]
#         b,c,l=features.shape
        b,l,c=features.shape
        ahat = rearrange(features,"b l c -> c (b l)")
        V = V + torch.mm(ahat,ahat.t())/(b*l)
        BATCH_NUM +=1
    V = V/BATCH_NUM

    w1, v1 = torch.linalg.eigh(V + torch.eye(BASIS1_NUM).to(device) * 1e-7, UPLO='U')
#     w1, v1 = torch.eig(V + torch.eye(BASIS1_NUM).to(device) * 1e-7, eigenvectors = True)
    whiteMat = torch.mm((w1.add(1e-3).pow(-0.5)).diag(), v1.t())
    colorMat = torch.mm(v1, w1.add(1e-3).pow(0.5).diag())

    feat_size = model.encoder.patch_embs.out_features
    seq_length = model.encoder.patch_embs.num_groups
    X = torch.zeros(50000,seq_length,feat_size)
    X_test = torch.zeros(10000,seq_length,feat_size)
#     y = torch.tensor(train_dataset_eval.targets)
#     y_test =  torch.tensor(val_dataset_eval.targets)

    for idx in tqdm(range(0,len(data_train),load_batch_size)):
        x = data_train[idx:idx+load_batch_size].to(device)
        features = model(x).detach()[:,1:,:]
        b,l,c = features.shape
        features_batch = rearrange(features,"b l c ->c (b l)")
        features_batch = torch.mm(whiteMat,features_batch)
        features = rearrange(features_batch,"c (b l) ->b l c", b = b)
        features = features.div(features.norm(dim=1,keepdim=True))
        X[idx:idx+load_batch_size,...] = features.cpu()

    for idx in tqdm(range(0,len(data_test),load_batch_size)):
        x = data_test[idx:idx+load_batch_size].to(device)
        features = model(x).detach()[:,1:,:]
        b,l,c = features.shape
        features_batch = rearrange(features,"b l c ->c (b l)")
        features_batch = torch.mm(whiteMat,features_batch)
        features = rearrange(features_batch,"c (b l) ->b l c", b = b)
        features = features.div(features.norm(dim=1,keepdim=True))
        X_test[idx:idx+load_batch_size,...] = features.cpu()
    knn_c = WeightedKNNClassifier(k=30,T=0.03)
    knn_c.update(X.flatten(1),train_label,X_test.flatten(1),test_label)
    acc2 = knn_c.compute()
    _ = model.train()
    del X_test,X
    gc.collect()
    
    return acc2[0]

def eval_model_KNN_(model,data_train,data_test,train_label,test_label,load_batch_size=400,BASIS1_NUM=192,perm=None,device = "cuda:0" ):
    device = model.encoder.patch_embs.weight.device
    _=model.eval()
    V = torch.zeros(BASIS1_NUM,BASIS1_NUM).to(device)
    BATCH_NUM = 0
    # for e in range(epochs):

    train_dl = data_train
    test_dl = data_test
    X_train_size = len(data_train)*load_batch_size
    X_test_size = len(data_test)*load_batch_size
        
    idx = 0
    for x,i in tqdm(train_dl):
        x=perm(x)
        x = x.to(device)
        idx +=load_batch_size
        features = model(x).detach()[:,1:,:]
        b,l,c=features.shape
        ahat = rearrange(features,"b l c -> c (b l)")
        V = V + torch.mm(ahat,ahat.t())/(b*l)
        BATCH_NUM +=1
    V = V/BATCH_NUM

    w1, v1 = torch.linalg.eigh(V + torch.eye(BASIS1_NUM).to(device) * 1e-7, UPLO='U')
    whiteMat = torch.mm((w1.add(1e-3).pow(-0.5)).diag(), v1.t())
    colorMat = torch.mm(v1, w1.add(1e-3).pow(0.5).diag())

    X = torch.zeros(X_train_size,l,c)
    X_test = torch.zeros(X_test_size,l,c)

    idx = 0
    for x,i in tqdm(train_dl):
        x=perm(x)
        x = x.to(device)
        features = model(x).detach()[:,1:,:]
        b,l,c = features.shape
        features_batch = rearrange(features,"b l c ->c (b l)")
        features_batch = torch.mm(whiteMat,features_batch)
        features = rearrange(features_batch,"c (b l) ->b l c", b = b)
        features = features.div(features.norm(dim=1,keepdim=True))
        X[idx:idx+load_batch_size,...] = features.cpu()
        idx +=load_batch_size
    idx = 0
    for x,i in tqdm(test_dl):
        x=perm(x)
        x = x.to(device)
        features = model(x).detach()[:,1:,:]
        b,l,c = features.shape
        features_batch = rearrange(features,"b l c ->c (b l)")
        features_batch = torch.mm(whiteMat,features_batch)
        features = rearrange(features_batch,"c (b l) ->b l c", b = b)
        features = features.div(features.norm(dim=1,keepdim=True))
        X_test[idx:idx+load_batch_size,...] = features.cpu()
        idx +=load_batch_size
    
    knn_c = WeightedKNNClassifier(k=30,T=0.03)
    knn_c.update(X.flatten(1),train_label,X_test.flatten(1),test_label)
    acc2 = knn_c.compute()
    _ = model.train()
    del X_test,X
    gc.collect()
    
    return acc2[0]