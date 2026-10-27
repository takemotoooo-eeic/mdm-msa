import hydra
from omegaconf import DictConfig

import mdmsa


@mdmsa.main()
def main(config: DictConfig) -> None:
    evaluator_config = config.train.evaluator
    evaluator = hydra.utils.instantiate(evaluator_config, config)
    evaluator.run()


if __name__ == "__main__":
    main()
