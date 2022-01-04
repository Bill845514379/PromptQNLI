cfg = {
    'gpu_id': 0,
    'max_len': 100,
    'train_batch_size': 1,
    'test_batch_size': 32,
    'learning_rate': 1e-5,
    'epoch': 20,
    'K': 16,
    'Kt': 2000,
    'template': '[X1] ? [MASK] , [X2]',
    'answer': ['no', 'yes'],
    'device': 'cuda',
    'optimizer': 'Adam',
    'word_size': 50265
}

hyper_roberta = {
    'word_dim': 1024,
    'dropout': 0.1
}

path = {
    'train_path': 'data/QNLI/train.tsv',
    'test_path': 'data/QNLI/test.tsv',
    'roberta_path': 'roberta-large'
}
