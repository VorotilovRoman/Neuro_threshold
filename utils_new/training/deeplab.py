"""Process entry point for deeplab training."""

if __name__ == "__main__" and not __package__:
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from utils_new.training.cli import semantic_main


def main(argv=None):
    return semantic_main("deeplab", argv)


if __name__ == "__main__":
    from utils_new.training.protocol import configure_worker_streams

    configure_worker_streams()
    raise SystemExit(main())
