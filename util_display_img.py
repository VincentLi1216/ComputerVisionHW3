import matplotlib.pyplot as plt


def plot(data, title, subplot_index, plot_size=(4,3)):
    plt.subplot(plot_size[0], plot_size[1], subplot_index)
    plt.imshow(data, cmap='gray')
    plt.title(title)