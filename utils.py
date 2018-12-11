import numpy as np
import matplotlib
matplotlib.use("Agg") # NEEDED TO SAVE TO MOVIE
import matplotlib.animation as manimation
import matplotlib.pyplot as plt

def cartesian_product(x,y):
    return np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])

def create_movie(data, path, plt_func=None, fps=1):
    """
    This function makes a rainbow movie.

    Args:
        data (array-like): A 3D numpy array, # of frames * X dim * Y dim
        path (str): Where the image is displayed; by default, it is drawn (note: draw means other subsequent plt commands will be drawn on top)
        plt_func (function): A function used for plotting the image; must take single X*Y frame as an input
        fps (int): Frames per second; higher frame rate, longer it takes to process; 18 is the max/native rate for 16x16 right now

    Returns:
        None

    """

    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib',
                    comment='Movie support!')
    writer = FFMpegWriter(fps=15, metadata=metadata)

    fig = plt.figure(1, facecolor='w')

    # Lower frame rate for faster processing
    factor=2
    end_limit = int(data.shape[0]/factor)
    print(end_limit)
    with writer.saving(fig, path, 100):
        for i in range(0, end_limit):
            plt_func(data[int(i * factor)])
            writer.grab_frame()
            plt.clf()

def one_hot(labels):
    labels = np.asarray(labels)
    b = np.zeros((labels.size, labels.max() + 1))
    b[np.arange(labels.size), labels] = 1
    return b