import os.path
import numpy as np
import argparse
import pandas as pd
import time
import pickle
import torch
from causallearn.search.ConstraintBased.PC import pc
from causal_tgan.model.dagTGAN import DagTGAN
from causal_tgan.helper.feature_info import FeatureINFO
from causal_tgan.helper.utils import data_transform, adjacency_to_parent_list
from causal_tgan.helper.trainer import train_full_knowledge
from causal_tgan.configuration import TrainingOptions, CausalTGANConfig
from causal_tgan.model.dagTGAN import load_model
from causal_tgan.helper.utils import restore_feature_info
from scipy.stats import rankdata
from sklearn.covariance import EmpiricalCovariance
import read_data_causal
import get_results

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="wine", type=str, nargs='?', help='dataset name')
parser.add_argument('--method', default="causal_ano", type=str, nargs='?', help='anomaly detection method')
parser.add_argument("--normalize", default="plain", type=str, nargs='?', help='numeric normalization')
parser.add_argument("--catencode", default="ordinal", type = str, nargs='?', help='categorical encoding')
parser.add_argument('--rl_lambda', default="0.01", type=str, nargs='?', help='weight of reinforcement learning')
parser.add_argument('--batchsize', default="64", type=str, nargs='?', help='batch size')
parser.add_argument('--score_normalize', default="True", type=str, nargs='?', help='normalize scores of root and child')
parser.add_argument('--runs', default="3", type=str, nargs='?', help='no of times to run algorithm')
parser.add_argument('--location', default="server", type=str, nargs='?', help='location to run algorithm')
args = parser.parse_args()
print("dataset: {}, method: {}, normalize: {}, categorical_encoding: {}".
      format(args.dataset, args.method, args.normalize, args.catencode))

if not torch.cuda.is_available():
    device = 'cpu'
else:
    device = 'cuda'
print("device: {}".format(device))

dataset_input = args.dataset
method_input = args.method
rl_lambda = float(args.rl_lambda)
batch_size = int(args.batchsize)
normalize = args.normalize # plain, ctgan
categorical_encode = args.catencode
score_normalize = args.score_normalize == "True"
n_run = int(args.runs)
location = args.location

if dataset_input == "all":
    datasets = ['breastw', 'ecoli', 'wine', 'annthyroid', 'mammography', 'pendigits', 'thyroid', 'waveform', 'wbc', 'shuttle',
                'cardio', 'glass', 'vowels', 'magic_gamma', 'pageblocks', 'stamps', 'yeast', 'imgseg',
                'lymphography', 'nhis', 'acd', 'spd', 'cmc', 'damre', 'os', 'smd', 'bank', 'seismic']
elif dataset_input == "odds":
    datasets = ['breastw', 'ecoli', 'wine', 'annthyroid', 'mammography', 'pendigits', 'thyroid', 'waveform', 'wbc', 'shuttle',
                'cardio', 'glass', 'vowels', 'magic_gamma', 'pageblocks', 'stamps', 'yeast', 'imgseg']
elif dataset_input == "mixed":
    datasets = ['lymphography', 'nhis', 'acd', 'spd', 'cmc', 'damre', 'os', 'smd', 'bank', 'seismic']
else:
    datasets = [dataset_input]
print("datasets: {}".format(datasets))

if method_input == "all":
    methods = ["causal_ano"]
else:
    methods = [method_input]
print("methods: {}".format(methods))

def normalize_score(score_train_raw, score_test_raw):
    """
    Convert raw scores to percentile ranks (0 to 1).
    Robust to outliers and scale differences.
    """
    # 1. Combine Train and Test to get a global reference for ranking
    combined_scores = np.concatenate([score_train_raw, score_test_raw])
    # 2. Compute Rank (1 = Lowest, N = Highest)
    ranks = rankdata(combined_scores, method='average')
    # 3. Scale to [0, 1]
    # Subtract 1 so range is 0 to (N-1), then divide by N-1
    normalized_ranks = (ranks - 1) / (len(combined_scores) - 1)
    # 4. Split back into Train and Test
    n_train = len(score_train_raw)
    score_test_norm = normalized_ranks[n_train:]

    return score_test_norm

lenc_dataset_method_run, f1_dataset_method_run, auc_dataset_method_run, runtime_dataset_method_run = [], [], [], []
for dataset in datasets:
    print("dataset: {}".format(dataset))
    lenc_method_run, f1_method_run, auc_method_run, runtime_method_run = [], [], [], []
    for method in methods:
        print("method: {}".format(method))
        lenc_run, f1_run, auc_run, runtime_run = np.zeros(n_run), np.zeros(n_run), np.zeros(n_run), np.zeros(n_run)
        for run in range(n_run):
            t_start = time.time()
            print("run: {}".format(run))
            file_name = ("ds_{}_{}_sn_{}_bs{}_r{}_{}".format(dataset, method, score_normalize, batch_size, run, location))
            np.random.seed(run)
            (X_train, X_test, y_train, y_test, n_train, n_test, n_feature, n_class,
             feature_names, categorical_columns, class_name) = (
                read_data_causal.load_data(dataset, n_splits = n_run, split_idx = run, seed=42))
            if isinstance(y_test, pd.Series):
                y_test = np.array(y_test)
            # transform dataframe to numpy array
            X = pd.concat([X_train, X_test], axis=0)
            X_np, feature_names = read_data_causal.df_to_numpy(X, cat_encode=categorical_encode)
            X_train = X_np[:n_train]
            X_test = X_np[n_train:]
            # need to re-construct feature_names as categorical_columns are moved to the end
            print("feature_names: {}".format(feature_names))

            if (os.path.exists("./results/{}/y_pred_{}.npz".format(method, file_name))):
                print("load predicted scores")
                y_pred = np.load("./results/{}/y_pred_{}.npz".format(method, file_name))
                weight_path = "./causal_models/{}_{}_{}_{}/".format(dataset, method, run, location)
                _, _, causal_list = restore_feature_info(weight_path)
                child_nodes = []
                for node in causal_list:
                    if len(node[1]) > 0:
                        child_nodes.append(node[0])
                lenc = len(child_nodes)
            else:
                print("save predicted scores")
                # run an anomaly detection method
                weight_path = "./causal_models/{}_{}_{}_{}/".format(dataset, "causal_ano", run, location)
                if os.path.exists(weight_path):
                    # loads model weights
                    print("loading model...{}".format(weight_path))
                    transformer, feature_info, causal_list = restore_feature_info(weight_path)
                    model = load_model(weight_path, feature_info, transformer)
                    if dataset == 'seismic' or dataset == 'bank':  # run PC faster
                        transformer_type = "ctgan"
                    else:
                        transformer_type = "plain"
                    print("transformer_type: {}".format(normalize))
                    transform_data, transformer, data_dims = data_transform(normalize, X_train,
                                                                            discrete_cols=categorical_columns)
                else:
                    os.mkdir(weight_path)
                    # save model weights
                    print("saving model...{}".format(weight_path))
                    X_train_df = pd.DataFrame(X_train)
                    X_train_df.columns = feature_names
                    # fix error: convert ndarray to list
                    feature_names = list([str(fea) for fea in feature_names])
                    # round continuous values
                    X_train_df = np.around(X_train_df, 2)
                    # run PC algorithm to find causal graph
                    print("using estimated causal graph")
                    pc_start = time.time()
                    lenc = 0
                    alphac = 0
                    # increase PC threshold until find child nodes
                    while lenc == 0 and alphac < 1:
                        alphac += 0.05
                        print("alphac: {:.2f}".format(alphac))
                        cg_org = pc(X_train_df.to_numpy(), alpha=alphac)
                        causal_graph = cg_org.G.graph
                        causal_graph[causal_graph == -1] = 0
                        causal_graph = np.transpose(causal_graph)
                        cg_org.G.graph = causal_graph
                        # convert causal graph in format of adjacency matrix to list of child-parent
                        causal_list = adjacency_to_parent_list(causal_graph, feature_names)
                        child_nodes = []
                        for node in causal_list:
                            if len(node[1]) > 0:
                                child_nodes.append(node[0])
                        lenc = len(child_nodes)
                    pc_end = time.time()
                    pc_run = (pc_end - pc_start) / 60
                    pc_run = round(pc_run, 2)
                    print("run PC for org data: {}".format(pc_run))
                    train_opts = TrainingOptions(batch_size=batch_size, runs_folder=weight_path)
                    gan_cfg = CausalTGANConfig(causal_graph=causal_list)
                    if dataset == 'seismic' or dataset == 'bank':  # run PC faster
                        transformer_type = "ctgan"
                    else:
                        transformer_type = "plain"
                    print("transformer_type: {}".format(normalize))
                    transform_data, transformer, data_dims = data_transform(normalize, X_train,
                                                                            discrete_cols=categorical_columns)
                    feat_info = FeatureINFO(feature_names, discrete_cols=categorical_columns, feature_dims=data_dims)
                    model = DagTGAN(gan_cfg, feat_info, transformer, ori_dag=cg_org.G, n_gen=n_train,
                                    rl_lambda=rl_lambda, pc_alpha=alphac)
                    train_full_knowledge(train_opts, transform_data, model, verbose=False)
                    # save model
                    with open(weight_path + 'options-and-config.pickle', 'wb+') as f:
                        pickle.dump(train_opts, f)  
                        pickle.dump(gan_cfg, f)
                    with open(weight_path + 'causal_graph.pickle', 'wb') as f:
                        pickle.dump(causal_list, f)
                    with open(weight_path + 'transformer.pickle', 'wb') as f:
                        pickle.dump(transformer, f)
                    with open(weight_path + 'featureInfo.pickle', 'wb') as f:
                        pickle.dump(feat_info, f)
                if model.causal_controller is not None:
                    model.causal_controller.set_causal_mechanisms_eval()
                child_nodes = []
                for node in causal_list:
                    if len(node[1]) > 0:
                        child_nodes.append(node[0])
                lenc = len(child_nodes)

                if method == "causal_ano":
                    X_test_transform, _ = transformer.transform(X_test)
                    X_test_transform = torch.tensor(X_test_transform.astype(np.float32), device=torch.device(device))
                    X_train_transform = torch.tensor(transform_data.astype(np.float32), device=torch.device(device))
                    # there is a causal graph
                    print("compute score_causal for all nodes")
                    feature_extractor = model.discriminator.model[:-1]
                    feature_extractor.eval()
                    X_train_feat = feature_extractor(X_train_transform)
                    X_test_feat = feature_extractor(X_test_transform)
                    robust_cov = EmpiricalCovariance(assume_centered=False)
                    robust_cov.fit(X_train_feat.detach().cpu().numpy())
                    score_causal = robust_cov.mahalanobis(X_test_feat.detach().cpu().numpy())
                    if score_normalize:
                        print("normalize score_causal")
                        score_causal_train = robust_cov.mahalanobis(X_train_feat.detach().cpu().numpy())
                        score_causal = normalize_score(score_causal_train, score_causal)
                    print("score_causal - min: {:.2f}, max: {:.2f}".format(score_causal.min(), score_causal.max()))
                    y_pred = score_causal

                # save result to numpy file
                with open("./results/{}/y_pred_{}.npz".format(method, file_name), "wb") as f:
                    np.save(f, y_pred)

            t_end = time.time()
            runtime = (t_end - t_start) / 60
            runtime = round(runtime, 2)
            ### from Ano-LLM code
            # y_test: data labels, 0 indicates normal samples and 1 indicates abnormal samples
            # y_pred: predicted anomaly scores, higher score indicates higher likelihoods to be anomaly
            acc, f1, auc = get_results.tabular_metrics(y_test, y_pred)
            acc, f1, auc = round(acc, 4), round(f1, 4), round(auc, 4)
            print("dataset: {}, method: {}, run: {}, acc: {}, f1: {}, auc: {}".format(dataset, method, run, acc, f1, auc))
            # save accuracy of each run of each method for each train_size in each dataset
            lenc_run[run], f1_run[run], auc_run[run], runtime_run[run] = lenc, f1, auc, runtime
            # save result to text file
            if run == (n_run - 1):
                with open('./results/{}/{}.txt'.format(method, file_name), 'w') as f:
                    lenc_avg = round(np.mean(lenc_run), 2)
                    f1_avg, f1_std = round(np.mean(f1_run), 4), round(np.std(f1_run), 4)
                    auc_avg, auc_std = round(np.mean(auc_run), 4), round(np.std(auc_run), 4)
                    runtime_avg, runtime_std = round(np.mean(runtime_run), 4), round(np.std(runtime_run), 4)
                    f.write("causal_list: {}\n".format(causal_list))
                    f.write("n_feature: {}\n".format(n_feature))
                    f.write("ALL lenc: {}, f1: {}, auc: {}\n". format(lenc_run, f1_run, auc_run))
                    f.write("AVG lenc: {}, f1: {} ({}), auc: {} ({})\n".format(lenc_avg, f1_avg, f1_std, auc_avg, auc_std))
                    f.write("ALL runtime: {}\n".format(runtime_run))
                    f.write("AVG runtime: {} ({}) (minutes)\n".format(runtime_avg, runtime_std))
        # save accuracy of n_run of each method in each dataset
        lenc_method_run.append(lenc_run)
        f1_method_run.append(f1_run)
        auc_method_run.append(auc_run)
        runtime_method_run.append(runtime_run)
    # save accuracy of n_run of all methods in each dataset
    lenc_dataset_method_run.append(lenc_method_run)
    f1_dataset_method_run.append(f1_method_run)
    auc_dataset_method_run.append(auc_method_run)
    runtime_dataset_method_run.append(runtime_method_run)

