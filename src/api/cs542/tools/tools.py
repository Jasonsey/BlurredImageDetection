from pathlib import Path


def init_path(paths: list):
    for path in paths:
        p = Path(path)
        if not p.exists():
            p.mkdir(parents=True)


if __name__ == '__main__':
    pass
