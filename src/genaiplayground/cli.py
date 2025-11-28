def main():
    from .extras.misc import is_env_enabled

    if is_env_enabled("USE_V1"):
        from .v1 import launcher
    else:
        from . import launcher

    launcher.launch()


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    main()
