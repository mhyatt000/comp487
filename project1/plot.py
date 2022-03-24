import matplotlib.pyplot as plt


def main():

    fig, ax = plt.subplots()

    data = {
        'vgg': [26.6e6, 0.9902, 138.81],
        'googlenet': [5.9e6, 0.9816, 261.6],
        'resnet': [11.1e6, 0.9890, 283.15],
        'densenet': [538000, 0.9911, 118.99],
        'senet(resnet)': [11.2e6, 0.9905, 295.9],
    }

    for m in data.keys():

        ax.scatter(data[m][2], data[m][1], label=m)

    ax.set(title='Accuracy vs Training Time for various CNN',
           xlabel='Time (sec)', ylabel='Accuracy (%)')
    ax.legend()
    plt.show()


if __name__ == '__main__':
    main()
