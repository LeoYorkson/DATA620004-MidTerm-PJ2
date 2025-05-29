from train import train


def main_pretrained():
    best_acc = 0
    for batch_size in [16, 32, 64]:
        for lr in [1e-5, 1e-4, 1e-3]:
            for lr_fc in [1e-3, 1e-2, 1e-1]:
                print("=" * 50)
                pretrained_config = {
                    'data_dir': 'data',
                    'pretrained': True,
                    'batch_size': batch_size,
                    'lr': lr,
                    'lr_fc': lr_fc,
                    'epochs': 30
                }
                print(batch_size, lr, lr_fc)
                best_acc = train(pretrained_config, best_acc)


def main_scratch():
    best_acc = 0
    for batch_size in [16, 32, 64]:
        for lr in [1e-5, 1e-4, 1e-3]:
            for lr_fc in [1e-3, 1e-2, 1e-1]:
                print("=" * 50)
                scratch_config = {
                    'data_dir': 'data',
                    'pretrained': False,
                    'batch_size': batch_size,
                    'lr': lr,
                    'lr_fc': lr_fc,
                    'epochs': 30
                }
                print(batch_size, lr, lr_fc)
                best_acc = train(scratch_config, best_acc)


if __name__ == '__main__':
    main_pretrained()
    main_scratch()
    
