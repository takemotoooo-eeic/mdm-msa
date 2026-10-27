from omegaconf import DictConfig

import exptemplete


@exptemplete.main(config_name="parse_run_command")
def main(config: DictConfig) -> None:
    """Return the command used to run recipe scripts."""
    print("python")


if __name__ == "__main__":
    main()
