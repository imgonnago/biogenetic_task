import numpy as np

#데이터셋을 훈련, 검증, 테스트로 나누는 함수. 데이터셋을 불러와서 인덱스를 섞은 후에 나눔.
def split_data(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    SNPDataset 또는 GMDataset을 받아서 데이터를 train/val/test로 split
    
    Args:
        dataset: SNPDataset 또는 GMDataset 인스턴스
        train_ratio, val_ratio, test_ratio: 분할 비율
        seed: random seed
    
    Returns:
        dict: {
            'train': {'data': ..., 'labels': ...},
            'val': {'data': ..., 'labels': ...},
            'test': {'data': ..., 'labels': ...}
        }
    """
    labels = dataset.labels
    num_samples = len(labels)
    idx = np.arange(num_samples)
    np.random.seed(seed)
    np.random.shuffle(idx)

    train_end = int(train_ratio * num_samples)
    val_end = train_end + int(val_ratio * num_samples)

    train_idx = idx[:train_end]
    val_idx = idx[train_end:val_end]
    test_idx = idx[val_end:]
    
    # SNPDataset인지 GMDataset인지 판별해서 데이터 추출
    if hasattr(dataset, 'SNP'):
        data = dataset.SNP
    else:
        data = dataset.gm
    
    # 데이터와 라벨 분할
    train_data = data[train_idx]
    val_data = data[val_idx]
    test_data = data[test_idx]
    
    train_labels = labels[train_idx]
    val_labels = labels[val_idx]
    test_labels = labels[test_idx]
    
    return {
        'train': {'data': train_data, 'labels': train_labels},
        'val': {'data': val_data, 'labels': val_labels},
        'test': {'data': test_data, 'labels': test_labels}
    }