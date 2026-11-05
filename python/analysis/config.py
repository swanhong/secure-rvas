from pathlib import Path
import tomllib


def load_party_config(directory: Path, party_id: int = 1) -> dict:
    config = {}
    for name in (
        "configGlobal.toml",
        f"configLocal.Party{party_id}.toml",
    ):
        with (directory / name).open("rb") as config_file:
            config.update(tomllib.load(config_file))
    return config


def load_site_configs(directory: Path, config_b: Path | None = None) -> list[dict]:
    config = load_party_config(directory)
    count = len(config["servers"]) - 1
    if count < 2 or set(config["servers"]) != {f"party{i}" for i in range(count + 1)}:
        raise ValueError("servers must contain P0 and contiguous data sites P1..PN")
    if config_b is not None and count != 2:
        raise ValueError("--config-b is only for two-site runs")
    sites = []
    for party in range(1, count + 1):
        source = config_b if party == 2 and config_b is not None else directory
        site = load_party_config(source, party)
        for name in ("servers", "chromosomes", "ancestries", "phenotype_columns", "num_cov"):
            if site[name] != config[name]:
                raise ValueError(f"P{party}: {name} differs from P1")
        sites.append(site)
    return sites


def site_input_paths(config: dict, ancestry: str, chromosome: int) -> dict[str, str]:
    values = dict(run_dir=config["run_dir"], ancestry=ancestry, chromosome=chromosome)
    return {
        name: str(Path(config[name].format(**values)).resolve())
        for name in ("genes_file", "keys_file", "genotype_dir", "covariate_file", "phenotype_file")
    }
