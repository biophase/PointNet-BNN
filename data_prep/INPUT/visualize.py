import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D     # registers the 3D projection


def plot(filename):
    # Load point cloud
    pt_cloud = np.load(filename)    # N x 3

    # Separate x, y, z coordinates
    xs = pt_cloud[:, 0]
    ys = pt_cloud[:, 1]
    zs = pt_cloud[:, 2]

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs, ys, zs)
    plt.show()

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
if __name__ == '__main__':
    import argparse

    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('filename', type=str, help='File name to point cloud')
    ARGS = PARSER.parse_args()

    FILENAME = ARGS.filename

    plot(FILENAME)
