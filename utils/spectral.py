import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.linalg import LinAlgError, qr, svd
from scipy.sparse import csc_matrix, coo_matrix
from sklearn.utils import check_random_state, as_float_array
import scipy
from model.sampling_retina_3 import cor_ls,corr_trans,img_size
from utils.others import area_std

def cluster_qr(vectors):
    k = vectors.shape[1]
    _, _, piv = qr(vectors.T, pivoting=True)
    ut, _, v = svd(vectors[piv[:k], :].T)
    vectors = abs(np.dot(vectors, np.dot(ut, v.conj())))
    return vectors.argmax(axis=1)


def discretize(
    vectors, *, copy=True, max_svd_restarts=30, n_iter_max=20, random_state=None
):
    random_state = check_random_state(random_state)

    vectors = as_float_array(vectors, copy=copy)

    eps = np.finfo(float).eps
    n_samples, n_components = vectors.shape

    # Normalize the eigenvectors to an equal length of a vector of ones.
    # Reorient the eigenvectors to point in the negative direction with respect
    # to the first element.  This may have to do with constraining the
    # eigenvectors to lie in a specific quadrant to make the discretization
    # search easier.
    norm_ones = np.sqrt(n_samples)
    for i in range(vectors.shape[1]):
        vectors[:, i] = (vectors[:, i] / np.linalg.norm(vectors[:, i])) * norm_ones
        if vectors[0, i] != 0:
            vectors[:, i] = -1 * vectors[:, i] * np.sign(vectors[0, i])

    # Normalize the rows of the eigenvectors.  Samples should lie on the unit
    # hypersphere centered at the origin.  This transforms the samples in the
    # embedding space to the space of partition matrices.
    vectors = vectors / np.sqrt((vectors**2).sum(axis=1))[:, np.newaxis]

    svd_restarts = 0
    has_converged = False

    # If there is an exception we try to randomize and rerun SVD again
    # do this max_svd_restarts times.
    while (svd_restarts < max_svd_restarts) and not has_converged:

        # Initialize first column of rotation matrix with a row of the
        # eigenvectors
        rotation = np.zeros((n_components, n_components))
        rotation[:, 0] = vectors[random_state.randint(n_samples), :].T

        # To initialize the rest of the rotation matrix, find the rows
        # of the eigenvectors that are as orthogonal to each other as
        # possible
        c = np.zeros(n_samples)
        for j in range(1, n_components):
            # Accumulate c to ensure row is as orthogonal as possible to
            # previous picks as well as current one
            c += np.abs(np.dot(vectors, rotation[:, j - 1]))
            rotation[:, j] = vectors[c.argmin(), :].T

        last_objective_value = 0.0
        n_iter = 0

        while not has_converged:
            n_iter += 1

            t_discrete = np.dot(vectors, rotation)

            labels = t_discrete.argmax(axis=1)
            vectors_discrete = csc_matrix(
                (np.ones(len(labels)), (np.arange(0, n_samples), labels)),
                shape=(n_samples, n_components),
            )

            t_svd = vectors_discrete.T * vectors

            try:
                U, S, Vh = np.linalg.svd(t_svd)
            except LinAlgError:
                svd_restarts += 1
                print("SVD did not converge, randomizing and trying again")
                break

            ncut_value = 2.0 * (n_samples - S.sum())
            if (abs(ncut_value - last_objective_value) < eps) or (n_iter > n_iter_max):
                has_converged = True
            else:
                # otherwise calculate rotation and continue
                last_objective_value = ncut_value
                rotation = np.dot(Vh.T, U.T)

    if not has_converged:
        raise LinAlgError("SVD did not converge")
    return labels

def distance_sklearn_metrics(d, k=5):
#     idx = np.argsort(d)[:,1:k+1]
    idx = np.argsort(d)[:,:k]
    d.sort()
#     d = -d[:,1:k+1]
    d = -d[:,:k]
    return d, idx


def adjacency(dist, idx):
    """Return adjacency matrix of a kNN graph"""
    M, k = dist.shape
    assert M, k == idx.shape
    # Weight matrix
    I = np.arange(0, M).repeat(k)
    J = idx.reshape(M*k)
    V = dist.reshape(M*k)
    W = scipy.sparse.coo_matrix((V, (I, J)), shape=(M, M))

    # No self-connections
    W.setdiag(0)

    # Undirected graph
    bigger = W.T > W
    W = W - W.multiply(bigger) + W.T.multiply(bigger)

    assert W.nnz % 2 == 0
    assert np.abs(W - W.T).mean() < 1e-10
    assert type(W) is scipy.sparse.csr.csr_matrix
    return W

def get_weighted_cluster(MI_score,K,oracle,w_factor,n_clusters,seed,label_file):
    MI_score = (MI_score+MI_score.T)/2
    K = K
    d,idx = distance_sklearn_metrics(-MI_score,K)
    graph = adjacency(d,idx)
    adj_matrix = np.array(graph.todense())
    # node_weight = np.array([(cor[2])**(oracle) for cor in cor_ls])
    cor_to_r = {}
    for cor in cor_ls:
        x,y,r = cor
        cor_to_r[(x,y)] = r
    r_ls_trans = []
    for i in range(len(corr_trans)):
        xy = corr_trans[i]
        x_off = int(xy//img_size)
        y_off = int(xy%img_size)
        r_ls_trans.append(cor_to_r[(x_off,y_off)])
    node_weight = np.array([r**(oracle) for r in r_ls_trans])

    print(node_weight)
    n_nodes = len(adj_matrix)
    degrees = adj_matrix.dot(np.ones(n_nodes))
    degrees_weight = adj_matrix.dot(np.ones(n_nodes))**w_factor


    laplacian = np.diag(degrees) - adj_matrix
    # laplacian = degree_matrix_inv_sqrt.dot(laplacian).dot(degree_matrix_inv_sqrt)
    node_weight_sqrt_inv  = np.diag(1/np.sqrt(node_weight))*(1/np.sqrt(degrees_weight))
    laplacian = node_weight_sqrt_inv.dot(laplacian).dot(node_weight_sqrt_inv)
    n_components = n_clusters+1
    eigenvalues, ev = eigsh(laplacian, n_components, which='SM')
    # embedding = ev[:,1:]
    embedding = node_weight_sqrt_inv @ ev[:,1:]
    #     embedding = degree_matrix_inv_sqrt @ ev[:,1:]
    labels = discretize(embedding, random_state=seed)
    print(len(np.unique(labels)))
    print(area_std(labels,cor_ls))
    np.save(label_file,labels)
    