def _get_version() -> str:
    from pathlib import Path

    import versioningit

    import nqcpresonator

    nqcpresonator_path = Path(nqcpresonator.__file__).parent
    return versioningit.get_version(project_dir=nqcpresonator_path.parent)


__version__ = _get_version()
