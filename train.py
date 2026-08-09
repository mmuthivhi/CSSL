import cssl
import hydra, argparse
from omegaconf import DictConfig, OmegaConf

from rich.console import Console
from rich.syntax import Syntax

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config : DictConfig):
    console = Console()

    yaml_str = OmegaConf.to_yaml(config)
    syntax = Syntax(yaml_str, "yaml", theme="monokai", line_numbers=True)
    console.print(syntax)

    trainer = cssl.Trainer(config)
    trainer.pretrain()
    


if __name__ == "__main__":
    main()

