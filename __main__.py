from src.screenshotter import screenshot, save, watch
from src.cli import get_args
from gooey import Gooey


@Gooey
def main():
    """
    execute script based on command line arguments
    """
    cli_args = get_args()
    should_watch = cli_args.watch
    bbox = cli_args.bbox
    if should_watch:
        mode = cli_args.mode
        stats = cli_args.stats
        watch(bbox=bbox, mode=mode, stats=stats)
    else:
        file_name = cli_args.file_name
        im = screenshot(bbox=bbox)
        save(im, file_name)


if __name__ == '__main__':
    main()
