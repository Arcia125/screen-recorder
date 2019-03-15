import argparse


def get_args():
    """
    get arguments from the command line
    """
    parser = argparse.ArgumentParser(
        description='Video or screenshotting tool')

    parser.add_argument('-n', dest='file_name',
                        help='set a file name', default='capture.png')
    parser.add_argument('-b', dest='bbox',
                        help='Define a bounding box', nargs=4, type=int)
    parser.add_argument('-w', dest='watch', help='Record and show part of the screen',
                        action='store_const', const='watch')
    parser.add_argument('--stats', dest='stats', help='show stats',
                        action='store_const', const='stats')
    mode_choices = ['rgb', 'edges', 'faces',
                    'blue', 'red', 'green', 'rank', 'median', 'laplacian']
    parser.add_argument('-m', dest='mode',
                        help='Video mode', default='rgb', choices=mode_choices)
    args = parser.parse_args()
    return args
