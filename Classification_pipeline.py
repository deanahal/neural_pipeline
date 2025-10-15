# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from processing.preprocessing import DataPreProcessing, feature_selection
from models.classifier import Classifier

# ------------------------------------------------------------
# Run Classification Pipeline
# ------------------------------------------------------------
def run_pipeline(neural_data, event_data, event_codes, event_names, trials_to_use,
                 bin_size=1, norm_method=None, feature_selection_method='lasso',
                 time_to_use=(0,1000), max_num_features=5, output_dim=2,
                 time_before_event=1000, time_after_event=3000,
                 classifier_name='svm', streams_to_use=['spikes','lfp','sbp'],
                 only_test=False, selected_features=None, expand_features=False, keep_entire_electrode=False):
    """
    Runs the full classification pipeline:
    - Locks neural data to events
    - Bins and normalizes data
    - Optionally expands features
    - Selects features and trains a classifier
    """
    feature_dict_list_spikes = []
    feature_dict_list_sbp = []
    feature_dict_list_lfp = []
    X_streams = []
    X_bands = []

    if only_test:
        assert selected_features is not None

    # ------------------ Data preprocessing ------------------
    data_preprocessing = DataPreProcessing(neural_data, event_data)
    neural_data_locked = data_preprocessing.lock_data_to_event(
        event_codes, event_names,
        bin_size=bin_size,
        time_before_event=time_before_event,
        time_after_event=time_after_event,
        trials_to_use=trials_to_use,
        norm_method=norm_method
    )

    # ------------------ Create samples per stream ------------------
    if 'spikes' in streams_to_use:
        X_spikes, y, feature_dict_list_spikes = data_preprocessing.create_samples(
            neural_data_locked=neural_data_locked,
            stream='spikes',
            time_to_use=time_to_use,
            output_dim=output_dim
        )
        X_streams.append(X_spikes)

    if 'sbp' in streams_to_use:
        X_sbp, y, feature_dict_list_sbp = data_preprocessing.create_samples(
            neural_data_locked=neural_data_locked,
            stream='sbp',
            time_to_use=time_to_use,
            output_dim=output_dim
        )
        X_streams.append(X_sbp)

    if 'lfp' in streams_to_use:
        X_lfp, y, feature_dict_list_lfp = data_preprocessing.create_samples(
            neural_data_locked=neural_data_locked,
            stream='lfp',
            time_to_use=time_to_use,
            output_dim=output_dim
        )
        X_streams.append(X_lfp)

    feature_dict_list = feature_dict_list_spikes + feature_dict_list_sbp + feature_dict_list_lfp

    # ------------------ Expand features if requested ------------------
    if expand_features:
        # expand LFP into band power
        X_a, y, feature_dict_list_a = data_preprocessing.create_samples(
            neural_data_locked=neural_data_locked, stream='alpha', time_to_use=time_to_use, output_dim=output_dim)
        X_bands.append(X_a)
        X_b, y, feature_dict_list_b = data_preprocessing.create_samples(
            neural_data_locked=neural_data_locked, stream='beta', time_to_use=time_to_use, output_dim=output_dim)
        X_bands.append(X_b)
        X_t, y, feature_dict_list_t = data_preprocessing.create_samples(
            neural_data_locked=neural_data_locked, stream='theta', time_to_use=time_to_use, output_dim=output_dim)
        X_bands.append(X_t)
        X_g, y, feature_dict_list_g = data_preprocessing.create_samples(
            neural_data_locked=neural_data_locked, stream='gamma', time_to_use=time_to_use, output_dim=output_dim)
        X_bands.append(X_g)

        X_bands = np.concatenate(X_bands, axis=1)
        feature_band_list = feature_dict_list_a + feature_dict_list_b + feature_dict_list_t + feature_dict_list_g

        # expand spikes into temporal features using PCA
        X_temporal_pca, y, feature_dict_list_temporal_pca = data_preprocessing.create_samples(
            neural_data_locked=neural_data_locked,
            stream='spikes_temporal_pca',
            time_to_use=time_to_use,
            output_dim=output_dim
        )
        X_temporal_pca = np.concatenate(X_temporal_pca, axis=1)

    print(f"number of features: {len(feature_dict_list)}")

    # ------------------ Concatenate streams ------------------
    X = np.concatenate(X_streams, axis=1)

    # Flatten 3D arrays if needed
    if len(X.shape) == 3:
        new_feature_dict_list = []
        n_samples, n_neurons, n_timepoints = X.shape
        X = X.reshape(n_samples, n_neurons * n_timepoints)
        for d in feature_dict_list:
            for i in range(n_timepoints):
                new_d = d.copy()
                new_d["time bin"] = i + 1
                new_feature_dict_list.append(new_d)

        if expand_features:
            n_samples, n_neurons, n_timepoints = X_bands.shape
            X_bands = X_bands.reshape(n_samples, n_neurons * n_timepoints)
            X = np.concatenate((X, X_bands), axis=1)

            for d in feature_band_list:
                for i in range(n_timepoints):
                    new_d = d.copy()
                    new_d["time bin"] = i + 1
                    new_feature_dict_list.append(new_d)

            _, components = X_temporal_pca.shape
            for d in feature_dict_list_temporal_pca:
                for i in range(components):
                    new_d = d.copy()
                    new_d["component"] = i + 1
                    new_feature_dict_list.append(new_d)

        feature_dict_list = new_feature_dict_list
    else:
        if expand_features:
            X = np.concatenate(X, np.concatenate(X_bands, X_temporal_pca, axis=1), axis=1)

    # ------------------ Scale features ------------------
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X[np.isnan(X)] = 0

    # ------------------ Feature selection ------------------
    if selected_features is not None:
        sorted_indices = selected_features
    else:
        sorted_indices = feature_selection(X=X, y=y, method=feature_selection_method,
                                           max_num_features=max_num_features, n_splits=5)

    if keep_entire_electrode and selected_features is None:
        new_feature_indices = []
        for feature in sorted_indices:
            electrode = feature_dict_list[feature]['electrode']
            stream = feature_dict_list[feature]['stream']
            new_feature_indices.append(np.where([(f['electrode'] == electrode) and (f['stream'] == stream)
                                                for f in feature_dict_list])[0])
        sorted_indices = np.unique(np.concatenate(new_feature_indices))

    if only_test:
        return X, y

    print(sorted_indices)

    # ------------------ Train classifier ------------------
    classifier = Classifier(X[:, sorted_indices], y, classifier=classifier_name, kernel='rbf', C=1, num_iterations=100, test_size=0.2)
    classifiers, accuracies = classifier.train()

    results_dict = {
        'classifier_name': classifier_name,
        'bin_size': bin_size,
        'norm_method': norm_method,
        'feature_selection_method': feature_selection_method,
        'time_to_use': time_to_use,
        'max_num_features': max_num_features,
        'sorted_indices': sorted_indices,
        'accuracies': np.array(accuracies)
    }

    return classifiers, accuracies, results_dict, X, y, sorted_indices, feature_dict_list

# ------------------------------------------------------------
# Reduced Dimensionality Visualization
# ------------------------------------------------------------
def draw_reduced_dim(X, y):
    """
    Draw PCA and t-SNE projections of X
    X: (n_samples, n_features)
    y: (n_samples,)
    """
    le = LabelEncoder()
    y = le.fit_transform(y)
    X = np.nan_to_num(X)

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # t-SNE
    tsne = TSNE(n_components=2, random_state=0, init='pca', perplexity=10)
    if np.all(X == 0):
        print("All features are zero. t-SNE cannot be applied.")
        return
    X_tsne = tsne.fit_transform(X)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(X_pca[y==0, 0], X_pca[y==0, 1], label='Class 0', alpha=0.7)
    axes[0].scatter(X_pca[y==1, 0], X_pca[y==1, 1], label='Class 1', alpha=0.7)
    axes[0].set_title('PCA Space')
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    axes[0].legend()

    axes[1].scatter(X_tsne[y==0, 0], X_tsne[y==0, 1], label='Class 0', alpha=0.7)
    axes[1].scatter(X_tsne[y==1, 0], X_tsne[y==1, 1], label='Class 1', alpha=0.7)
    axes[1].set_title('t-SNE Space')
    axes[1].set_xlabel('Dim 1')
    axes[1].set_ylabel('Dim 2')
    axes[1].legend()

    plt.tight_layout()
    plt.show()
