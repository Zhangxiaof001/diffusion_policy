"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf
import pathlib
from diffusion_policy.workspace.base_workspace import BaseWorkspace

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,  # 版本基础，设置为 None
    config_path=str(pathlib.Path(__file__).parent.joinpath(  # 配置文件路径
        'diffusion_policy','config'))
)
def main(cfg: OmegaConf):
    # 立即解析配置，确保所有 ${now:} 解析器使用相同的时间
    OmegaConf.resolve(cfg)

    # 获取配置中指定的类
    cls = hydra.utils.get_class(cfg._target_)

    # 创建工作空间实例
    workspace: BaseWorkspace = cls(cfg)

    # 运行工作空间
    workspace.run()

if __name__ == "__main__":
    main()
