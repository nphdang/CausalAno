import os
import numpy as np
import pandas as pd
import copy
import scipy.io
import ucimlrepo
# Preprocessing
import string
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from feature_engine.encoding import RareLabelEncoder

MIXED = ['lymphography', 'nhis', 'acd', 'spd', 'cmc', 'damre', 'os', 'smd', 'bank', 'seismic']
ODDS = ['breastw', 'ecoli', 'wine', 'annthyroid', 'mammography', 'pendigits', 'thyroid', 'waveform', 'wbc', 'shuttle',
        'cardio', 'glass', 'vowels', 'magic_gamma', 'pageblocks', 'stamps', 'yeast', 'imgseg']

# Map of dataset names to their corresponding dataset IDs in the UCI ML repository
DATA_MAP ={
	# ucimlrepo
	'breastw':15,
	'cardio':193,
	'ecoli': 39,
	'lymphography': 63,
	'wine': 109,
	'yeast':110,
	# csv files
    'seismic': None,
    'nhis': None,
    'acd': None,
    'spd': None,
    'cmc': None,
    'bank': None,
    'smd': None,
    'damre': None,
    'os': None,
	# adbench datasets
	'annthyroid': 2,
    'glass': 14,
    'mammography': 23,
    'pendigits':28,
    'shuttle':32,
    'thyroid':38,
    'vowels':40,
    'magic_gamma': 22,
    'pageblocks': 27,
    'stamps': 37,
    'waveform': 41,
    'wbc': 42,
    # drl datasets
    'imgseg': None,
}

def convert_np_to_df(X_np):
	n_train, n_cols = X_np.shape
	# Add missing column names
	L = list(string.ascii_uppercase) + [letter1+letter2 for letter1 in string.ascii_uppercase for letter2 in string.ascii_uppercase]
	columns = [L[i] for i in range(n_cols)]
	df = pd.DataFrame(data = X_np, columns = columns)

	return df

def load_dataset(dataset_name):
    if os.path.exists("./data/{}_new.csv".format(dataset_name)):
        print("load data from csv file")
        df = pd.read_csv('./data/{}_new.csv'.format(dataset_name))
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
    else:
        print("save data to csv file")
        if dataset_name == 'wine':
            dataset_id = DATA_MAP[dataset_name]
            df = ucimlrepo.fetch_ucirepo(id=dataset_id).data['original']
            columns = [name.replace('_', ' ') for name in df.columns[:-1]]
            df.columns = columns + [df.columns[-1]]
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
            # map labels to indices
            y = y - 1
        elif dataset_name == 'breastw':
            dataset_id = DATA_MAP[dataset_name]
            df = ucimlrepo.fetch_ucirepo(id=dataset_id).data['original']
            columns = [name.replace('_', ' ') for name in df.columns[:-1]]
            df.columns = columns + [df.columns[-1]]
            X = df.iloc[:, 1:-1]
            y = df.iloc[:, -1]
            # map labels to indices
            y[y == 2] = 0
            y[y == 4] = 1
        elif dataset_name == 'cardio':
            dataset_id = DATA_MAP[dataset_name]
            uci_dataset = ucimlrepo.fetch_ucirepo(id=dataset_id)
            # get columns descriptions
            var_info = uci_dataset['metadata']['additional_info']['variable_info']
            L = [k.split(' - ') for k in var_info.split('\n')]
            column_dict = {}
            for k, v in L:
                column_dict[k] = v.strip('\r')
            df = uci_dataset.data['original']
            df = df[df['NSP'] != 2].reset_index(drop=True)
            y = df['NSP'].map({3: 1, 1: 0})  # map pathologic to 1, normal to 0
            y = y.to_numpy()
            df.drop(['CLASS', 'NSP'], inplace=True, axis=1)
            new_columns = [column_dict[c] for c in df.columns]
            df.columns = new_columns
            X = df
        elif dataset_name == 'ecoli':
            dataset_id = DATA_MAP[dataset_name]
            uci_dataset = ucimlrepo.fetch_ucirepo(id=dataset_id)
            columns = uci_dataset['variables']['description'][:8]
            X = uci_dataset.data['original'].drop(['class'], axis = 1)
            X.columns = columns
            X = X.drop(X.columns[0], axis=1) # drop id column
            y = uci_dataset.data['original']['class'].map({'omL':1,'imL':1,'imS':1, 'cp':0, 'im':0, 'pp':0, 'imU':0, 'om':0})
            y = y.to_numpy()
        elif dataset_name == 'yeast':
            # the split is different from the one in the ADbench
            dataset_id = DATA_MAP[dataset_name]
            uci_dataset = ucimlrepo.fetch_ucirepo(id=dataset_id)
            df = uci_dataset.data['original']
            columns = [s.rstrip('.') for s in uci_dataset['variables']['description'][1:9]]
            y = df['localization_site'].map(
                {'CYT': 0, 'NUC': 0, 'MIT': 0, 'ME3': 0, 'ME2': 1, 'ME1': 1, 'EXC': 0, 'VAC': 0, 'POX': 0, 'ERL': 0})
            y = y.to_numpy()
            df.drop('localization_site', inplace=True, axis=1)
            df.drop('Sequence_Name', inplace=True, axis=1)
            df.columns = columns
            X = df
        elif dataset_name == 'lymphography':
            dataset_id = DATA_MAP[dataset_name]
            uci_dataset = ucimlrepo.fetch_ucirepo(id=dataset_id)
            df = uci_dataset.data['original']
            y = df['class'].map({1:1,2:0,3:0,4:1}) # 142 normal, 6 anomalies
            y = y.to_numpy()
            df.drop('class', inplace = True, axis = 1)
            df.drop('no. of nodes in', inplace = True, axis = 1)
            df['lymphatics'] = df['lymphatics'].map({1:'normal', 2:'arched', 3:'deformed', 4:'displaced'}).astype('object')
            df['defect in node'] = df['defect in node'].map({1:'no',2:'lacunar', 3:'lac. marginal', 4:'lac. central'}).astype('object')
            df['changes in lym'] = df['changes in lym'].map({1:'bean',2:'oval', 3:'round'}).astype('object')
            df['changes in node'] = df['changes in node'].map({1:'no',2:'lacunar', 3:'lac. marginal', 4:'lac. central'}).astype('object')
            df['changes in stru'] = df['changes in stru'].map({1:'no',2:'grainy', 3:'drop-like', 4:'coarse', 5:'diluted', 6: 'reticular', 7:'stripped', 8:'faint'}).astype('object')
            df['special forms'] = df['special forms'].map({1:'no',2:'chalices', 3:'vesicles'}).astype('object')
            for k in ['block of affere', 'bl. of lymph. c', 'bl. of lymph. s', 'by pass', 'extravasates', 'regeneration of', 'early uptake in', 'dislocation of', 'exclusion of no']:
                df[k] = df[k].map({1:'no',2:'yes'}).astype('object')
            X = df
        elif dataset_name == 'seismic':
            # https://archive.ics.uci.edu/ml/machine-learning-databases/00266/seismic-bumps.arff
            data_path = './data/seismic-bumps.arff'
            data, meta = scipy.io.arff.loadarff(data_path)
            df = pd.DataFrame(data)
            column_replacement = {
                'seismic': 'result of shift seismic hazard assessment in the mine working obtained by the seismic method',
                'seismoacoustic': 'result of shift seismic hazard assessment in the mine working obtained by the seismoacoustic method',
                'shift': 'information about type of a shift',
                'genergy': 'seismic energy recorded within previous shift by the most active geophone (GMax) out of geophones monitoring the longwall',
                'gpuls': 'a number of pulses recorded within previous shift by GMax',
                'gdenergy': 'a deviation of energy recorded within previous shift by GMax from average energy recorded during eight previous shifts',
                'gdpuls': 'a deviation of a number of pulses recorded within previous shift by GMax from average number of pulses recorded during eight previous shifts',
                'ghazard': 'result of shift seismic hazard assessment in the mine working obtained by the seismoacoustic method based on registration coming from GMax only',
                'nbumps': 'the number of seismic bumps recorded within previous shift',
                'nbumps2': 'the number of seismic bumps (in energy range [10^2,10^3)) registered within previous shift',
                'nbumps3': 'the number of seismic bumps (in energy range [10^3,10^4)) registered within previous shift',
                'nbumps4': 'the number of seismic bumps (in energy range [10^4,10^5)) registered within previous shift',
                'nbumps5': 'the number of seismic bumps (in energy range [10^5,10^6)) registered within the last shift',
                'nbumps6': 'the number of seismic bumps (in energy range [10^6,10^7)) registered within previous shift',
                'nbumps7': 'the number of seismic bumps (in energy range [10^7,10^8)) registered within previous shift',
                'nbumps89': 'the number of seismic bumps (in energy range [10^8,10^10)) registered within previous shift',
                'energy': 'total energy of seismic bumps registered within previous shift',
                'maxenergy': 'the maximum energy of the seismic bumps registered within previous shift',
            }
            # take log on magnitude columns
            df['maxenergy'] = np.log(df['maxenergy'].replace(0, 1e-6))
            df['energy'] = np.log(df['energy'].replace(0, 1e-6))
            # Rename the columns
            df.rename(columns=column_replacement, inplace=True)
            # Replace categorical values in the columns
            df['result of shift seismic hazard assessment in the mine working obtained by the seismic method'] = df['result of shift seismic hazard assessment in the mine working obtained by the seismic method'].replace({b'a': 'lack of hazard', b'b': 'low hazard', b'c': 'high hazard', b'd': 'danger state'})
            df['result of shift seismic hazard assessment in the mine working obtained by the seismoacoustic method'] = df['result of shift seismic hazard assessment in the mine working obtained by the seismoacoustic method'].replace({b'a': 'lack of hazard', b'b': 'low hazard', b'c': 'high hazard', b'd': 'danger state'})
            df['result of shift seismic hazard assessment in the mine working obtained by the seismoacoustic method based on registration coming from GMax only'] = \
                df['result of shift seismic hazard assessment in the mine working obtained by the seismoacoustic method based on registration coming from GMax only'].replace({b'a': 'lack of hazard', b'b': 'low hazard', b'c': 'high hazard', b'd': 'danger state'})
            df['information about type of a shift'] = df['information about type of a shift'].replace({'W': 'coal-getting', 'N': 'preparation shift'})
            y = df['class'].map({b'0':0,b'1':1})
            y = y.to_numpy()
            df.drop('class', inplace = True, axis = 1)
            X = df
        elif dataset_name == 'nhis':
            # https://www.kaggle.com/datasets/bonifacechosen/nhis-healthcare-claims-and-fraud-dataset?select=cleaned_nhis_with_fraud_types+%281%29.csv
            df = pd.read_csv('./data/nhis.csv')
            df.drop(columns=["Patient ID", "DATE OF ENCOUNTER", "DATE OF DISCHARGE"], inplace=True)
            df['FRAUD_TYPE'] = np.where(df['FRAUD_TYPE'].str.strip().eq('No Fraud'), 0, 1).astype(int)
            y = df['FRAUD_TYPE']
            y = y.to_numpy()
            df.drop('FRAUD_TYPE', inplace=True, axis=1)
            X = df
        elif dataset_name == 'acd':
            # https://www.kaggle.com/datasets/fcwebdev/synthetic-cybersecurity-logs-for-anomaly-detection
            df = pd.read_csv('./data/advanced_cybersecurity_data.csv')
            df.drop(columns=["Timestamp", "IP_Address", "Session_ID"], inplace=True)
            df['Status_Code'] = df['Status_Code'].astype(object)
            y = df['Anomaly_Flag']
            y = y.to_numpy()
            df.drop('Anomaly_Flag', inplace=True, axis=1)
            X = df
        elif dataset_name == 'spd':
            # https://www.kaggle.com/datasets/krishna1502/pegasus-spyware-attacksynthetic-dataset/data
            df = pd.read_csv('./data/synthetic_pegasus_dataset.csv')
            df.drop(columns=["user_id", "timestamp", "source_ip", "destination_ip"], inplace=True)
            df['anomaly_detected'] = np.where(df['anomaly_detected'].isnull(), 0, 1)
            df['ioc'] = df['ioc'].fillna('None')
            df['event_description'] = (df['event_description'].str.replace(r'^Event\s*', '', regex=True).astype(int))
            y = df['anomaly_detected']
            y = y.to_numpy()
            df.drop('anomaly_detected', inplace=True, axis=1)
            X = df
            X['event_description'] = X['event_description'].astype('int16')
        elif dataset_name == "cmc":
            # https://github.com/mala-lab/ADBenchmarks-anomaly-detection-datasets/tree/main/categorical%20data
            df = pd.read_csv('./data/cmc.csv')
            y = df['class_numberofchildren']
            y = y.to_numpy()
            df.drop('class_numberofchildren', inplace=True, axis=1)
            X = df
        elif dataset_name == "bank":
            # https://github.com/mala-lab/ADBenchmarks-anomaly-detection-datasets/tree/main/categorical%20data
            df = pd.read_csv('./data/bank.csv')
            y = df['class'].map({"yes":1, "no":0})
            y = y.to_numpy()
            df.drop('class', inplace=True, axis=1)
            X = df
        elif dataset_name == 'smd':
            # https://www.kaggle.com/datasets/ziya07/smart-meter-electricity-consumption-dataset/data
            df = pd.read_csv('./data/smart_meter.csv')
            df["Timestamp"] = pd.to_datetime(df["Timestamp"])
            df["hour_group"] = pd.cut(
                df["Timestamp"].dt.hour,
                bins=[-1, 6, 12, 18, 24],
                labels=["00-06", "06-12", "12-18", "18-24"]
            ).astype("category")
            df['is_weekend'] = (df["Timestamp"].dt.dayofweek >= 5).astype("category")
            df['is_weekend'] = df['is_weekend'].map({True: 'yes', False: 'no'})
            df["Anomaly_Label"] = df["Anomaly_Label"].map({'Normal': 0, 'Abnormal': 1})
            df.drop(columns=["Timestamp"], inplace=True)
            y = df['Anomaly_Label']
            y = y.to_numpy()
            df.drop('Anomaly_Label', inplace=True, axis=1)
            X = df
            numeric_data = X.select_dtypes(include=['float64', 'int64', 'uint8', 'int16', 'float32'])
            numeric_columns = numeric_data.columns.tolist()
            X[numeric_columns] = np.around(X[numeric_columns], 4)
        elif dataset_name == 'damre':
            # https://www.kaggle.com/datasets/iniyansel/insurance-reports-for-fraud-detection-training
            df = pd.read_csv('./data/damage_reports.csv')
            df["fraud"] = df["fraud"].map({False: 0, True: 1})
            df.drop(columns=["contract_date", "damage_id", "damage_date", 'policyholder'], inplace=True)
            y = df['fraud']
            y = y.to_numpy()
            df.drop('fraud', inplace=True, axis=1)
            X = df
        elif dataset_name == 'os':
            # https://www.kaggle.com/datasets/ziya07/os-kernel-anomaly-dataset?utm_source=chatgpt.com
            df = pd.read_csv('./data/os_kernel_power.csv')
            df["Label"] = df["Label"].map({"Normal": 0, "Anomaly": 1})
            df.drop(columns=["Timestamp", "PID"], inplace=True)
            y = df['Label']
            y = y.to_numpy()
            df.drop('Label', inplace=True, axis=1)
            X = df
        elif dataset_name in DATA_MAP.keys():
            dataset_root = "./data/"
            n = DATA_MAP[dataset_name]
            # datasets from ADBench
            if n is not None:
                for npz_file in os.listdir(dataset_root):
                    if npz_file.startswith(str(n) + '_' + dataset_name):
                        print(dataset_name, npz_file)
                        data = np.load(dataset_root + npz_file, allow_pickle=False)
                        break
                else:
                    ValueError('{} is not found.'.format(dataset_name))
            # datasets from DRL
            else:
                for npz_file in os.listdir(dataset_root):
                    if npz_file.startswith(dataset_name):
                        print(dataset_name, npz_file)
                        data = np.load(dataset_root + npz_file, allow_pickle=False)
                        break
                else:
                    ValueError('{} is not found.'.format(dataset_name))
            X_np, y = data['X'], data['y']
            X = convert_np_to_df(X_np)

        # save data to csv file
        if isinstance(y, pd.Series):
            y = np.array(y)
        X_y = np.append(X, y.reshape(-1, 1), axis=1)
        X_y_df = pd.DataFrame(X_y)
        X_y_df.columns = np.append(X.columns, "target")
        X_y_df.to_csv("./data/{}_new.csv".format(dataset_name), index=False)
	
    return X, y

def split_data(y: np.ndarray, n_splits: int = 5, train_ratio: float = 0.5, seed: int = 42):
    np.random.seed(seed)
    train_indices, test_indices = [], []
    for i in range(n_splits):
        normal_data_indices = np.where(y == 0)[0]
        abnormal_data_indices = np.where(y == 1)[0]
        data_length = len(normal_data_indices)
        index = np.random.permutation(normal_data_indices)
        train_index = index[:int(train_ratio * data_length)]
        test_index = index[int(train_ratio * data_length):]
        test_index = np.concatenate([test_index, abnormal_data_indices])
        train_index = np.random.permutation(train_index)
        test_index = np.random.permutation(test_index)
        train_indices.append(train_index)
        test_indices.append(test_index)

    return train_indices, test_indices

def normalize(X, normalize_method, n_buckets):
    # normalize_method: ['quantile', 'equal_width', 'language', 'none', 'standard']
    # n_buckets: 0-100
    X = copy.deepcopy(X)

    def ordinal(n):
        if np.isnan(n):
            return 'NaN'
        n = int(n)
        if 10 <= n % 100 <= 20:
            suffix = 'th'
        else:
            suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')

        return 'the ' + str(n) + suffix + ' percentile'

    word_list = ['Minimal', 'Slight', 'Moderate', 'Noticeable', 'Considerable', 'Significant', 'Substantial', 'Major',
                 'Extensive', 'Maximum']

    def get_word(n):
        n = int(n)
        if n == 10:
            return word_list[-1]

        return word_list[n]

    if normalize_method == 'quantile':
        for column in X.columns:
            if X[column].dtype in ['float64', 'int64', 'uint8', 'int16'] and X[column].nunique() > 1:
                ranks = X[column].rank(method='min')
                X[column] = ranks / len(X[column]) * 100
                X[column] = X[column].apply(ordinal)
    elif normalize_method == 'equal_width':
        for column in X.columns:
            if X[column].dtype in ['float64', 'int64', 'uint8', 'int16']:
                if X[column].nunique() > 1:
                    X[column] = X[column].astype('float64')
                    X[column] = (X[column] - X[column].min()) / (X[column].max() - X[column].min()) * n_buckets
                if 10 % n_buckets == 0:
                    X[column] = X[column].round(0) / 10
                    X[column] = X[column].round(1)
                else:
                    X[column] = X[column].round(0) / 100
                    X[column] = X[column].round(2)
    elif normalize_method == 'standard':
        for column in X.columns:
            if X[column].dtype in ['float64', 'int64', 'uint8', 'int16']:
                scaler = StandardScaler()
                scaler.fit(X[column].values.reshape(-1, 1))
                X[column] = scaler.transform(X[column].values.reshape(-1, 1))
                X[column] = X[column].round(1) # single-digit decimals
    elif normalize_method == 'language':
        for column in X.columns:
            if X[column].dtype in ['float64', 'int64', 'uint8', 'int16'] and X[column].nunique() > 1:
                X[column] = X[column].astype('float64')
                X[column] = (X[column] - X[column].min()) / (X[column].max() - X[column].min()) * 10
                X[column] = X[column].apply(get_word)
    else:
        raise ValueError('Invalid method. Choose either percentile, language or decimal')

    return X

def df_to_numpy(X: pd.DataFrame, cat_encode: str = 'ordinal', normalize_numbers: str = 'none'):
    numeric_data = X.select_dtypes(include=['float64', 'int64', 'uint8', 'int16', 'float32'])
    numeric_columns = numeric_data.columns.tolist()
    categorical_data = X.select_dtypes(include=['object', 'category'])
    categorical_columns = categorical_data.columns.tolist()
    print("numeric_columns: {}".format(numeric_columns))
    print("categorical_columns: {}".format(categorical_columns))

    if len(numeric_columns) > 0:
        for numeric_col in numeric_columns:
            # fill na
            X[numeric_col] = X[numeric_col].fillna(X[numeric_col].mean())
        if normalize_numbers == "minmax":
            print("scale numeric columns to min-max")
            # normalize it to [0, 1]
            X[numeric_columns] = MinMaxScaler().fit_transform(X[numeric_columns])
        elif normalize_numbers == "standard":
            print("standardize numeric columns")
            # normalize it to have zero mean and unit variance
            X[numeric_columns] = StandardScaler().fit_transform(X[numeric_columns])
    if len(categorical_columns) > 0:
        # categorical features:
        # group categories with low frequency into a single category
        encoder = RareLabelEncoder(
            tol=0.01,  # Minimum frequency to be considered as a separate class
            max_n_categories=None,  # Maximum number of categories to keep
            replace_with='Rare',  # Value to replace rare categories with
            variables=categorical_columns,  # Columns to encode
            missing_values='ignore',
        )
        X = encoder.fit_transform(X)
        # remove categories that have only one value
        for column in categorical_columns:
            if X[column].nunique() == 1:
                X.drop(column, inplace=True, axis=1)
        if cat_encode == 'ordinal':
            print("use ordinal encoding for categorical columns")
            le = LabelEncoder()
            for i in categorical_data.columns:
                categorical_data[i] = le.fit_transform(categorical_data[i])
        elif cat_encode == 'one_hot':
            print("use one_hot encoding for categorical columns")
            enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first')
            one_hot_encoded = enc.fit_transform(X[categorical_columns])
            categorical_data = pd.DataFrame(one_hot_encoded, columns=enc.get_feature_names_out(categorical_columns))
        else:
            raise ValueError('Invalid method. Choose either ordinal or one_hot')
        X_prime = X.drop(categorical_columns, axis=1)
        X = pd.concat([X_prime, categorical_data], axis=1)
    feature_names = X.columns
    X_np = X.to_numpy()
    print("X_np: {}".format(X_np.shape))

    return X_np, feature_names

def load_data(dataset="wine", binning='none', n_buckets=10, remove_feature_name=False,
              n_splits = 5, split_idx = 0, train_ratio = 0.5, seed = 42):
    if dataset == "synthetic":
        X_train = pd.read_csv('./data/X_train_syn.csv')
        X_test = pd.read_csv('./data/X_test_syn.csv')
        n_anomaly = 1000
        y_train = np.zeros(len(X_train), dtype=int)
        y_test = np.append(np.ones(n_anomaly, dtype=int), np.zeros(len(X_test) - n_anomaly, dtype=int))
    else:
        X, y = load_dataset(dataset)
        if binning != 'none':
            X = normalize(X, binning, n_buckets)
        if remove_feature_name == True:
            print("Removing column names")
            X.columns = [f"X{i+1}" for i in range(len(X.columns))]
        train_indices, test_indices = split_data(y, n_splits, train_ratio, seed)
        train_index, test_index = train_indices[split_idx], test_indices[split_idx]
        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y[train_index], y[test_index]
    n_train, n_test, n_feature, n_class = X_train.shape[0], X_test.shape[0], X_train.shape[1], len(np.unique(y_train))
    feature_names, class_name = X_train.columns, "target"
    categorical_columns = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    print("X_train: {}, y_train: {}".format(X_train.shape, y_train.shape))
    print("X_test: {}, y_test: {}".format(X_test.shape, y_test.shape))
    print("n_train: {}, n_test: {}, n_feature: {}, n_class: {}".format(n_train, n_test, n_feature, n_class))
    print("feature_names: {}, class_name: {}".format(feature_names, class_name))
    print("categorical_names: {}".format(categorical_columns))
    print("n_anomaly: {} ({}%)".format(np.sum(y_test), round(np.sum(y_test) / (n_train + n_test) * 100)))
	
    return X_train, X_test, y_train, y_test, n_train, n_test, n_feature, n_class, feature_names, categorical_columns, class_name

