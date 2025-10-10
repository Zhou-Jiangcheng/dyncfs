import imageio.v3 as iio


def images_to_gif(image_files, output_file, duration=0.2, loop=0):
    """
    Combine multiple images into a GIF animation.

    Parameters
    ----------
    image_files : list of str
        List of image file paths (in playback order).
    output_file : str
        Output GIF filename, e.g. 'output.gif'.
    duration : float
        Duration of each frame in seconds (default: 0.2).
    loop : int
        Number of loops, 0 means infinite loop.
    """
    # Read images and save as GIF
    frames = [iio.imread(img) for img in image_files]
    iio.imwrite(output_file, frames, duration=duration, loop=loop)
