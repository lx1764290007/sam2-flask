import sys
sys.path.append(".")  # 如果你用 VSCode/IDE 运行可能需要加这行

from hydra import initialize, compose

with initialize(config_path="pkg://sam2_configs", job_name="test"):
    cfg = compose(config_name="yaml")
    from omegaconf import OmegaConf

    print(OmegaConf.to_yaml(cfg))