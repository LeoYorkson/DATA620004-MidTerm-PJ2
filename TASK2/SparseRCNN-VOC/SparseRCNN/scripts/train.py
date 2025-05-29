import os
from mmengine import Config
from mmengine.runner import Runner

def main():
    cfg_path = os.path.join(os.path.dirname(__file__), '../configs/sparse_rcnn_voc.py')
    cfg = Config.fromfile(cfg_path)
    cfg.work_dir = '../work_dirs/sparse_rcnn_voc'
    os.makedirs(cfg.work_dir, exist_ok=True)
    runner = Runner.from_cfg(cfg)
    runner.train()

if __name__ == '__main__':
    main()
