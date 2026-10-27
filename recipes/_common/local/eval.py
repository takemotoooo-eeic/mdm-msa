import hydra
from omegaconf import DictConfig

import exptemplete


@exptemplete.main()
def main(config: DictConfig) -> None:
    evaluator_config = config.train.evaluator
    evaluator = hydra.utils.instantiate(evaluator_config, config)
    evaluator.run()


if __name__ == "__main__":
    main()
