import numpy as np
from skimage import morphology


def generate_sample(n_objects, size=512, obj_size=8, ndim=2):
    if isinstance(size, int):
        size = [size] * ndim

    sample = np.zeros(shape=size, dtype=bool)
    sample_pos_axis = np.unravel_index(
        np.random.randint(low=0, high=sample.size - 1, size=(n_objects,)),
        shape=sample.shape
    )

    sample[sample_pos_axis] = True

    if ndim == 2:
        se = morphology.disk(radius=obj_size // 2)
    else:
        se = morphology.ball(radius=obj_size // 2)

    morphology.dilation(sample, footprint=se, out=sample)

    return sample


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    sample_2d = generate_sample(n_objects=128, size=512, obj_size=27, ndim=2)

    plt.imshow(sample_2d)
    plt.show()

    sample_3d = generate_sample(n_objects=16, size=[64, 64, 64],
                                obj_size=17,
                                ndim=3)

    rr = 5
    cc = 5
    fig, axs = plt.subplots(rr, cc)
    z_offset = np.random.randint(low=0, high=sample_3d.shape[0] - rr * cc)
    for i, ax_i in zip(range(z_offset, z_offset + rr * cc), axs.ravel()):
        ax_i.imshow(sample_3d[i])

    plt.show()
