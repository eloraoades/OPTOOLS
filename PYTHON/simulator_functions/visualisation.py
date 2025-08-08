import matplotlib.pyplot as plt

def custom_plots_surf(ax, im, iskm=1, alongT_isY=1, labelcb=None):
    if alongT_isY:
        xlb = 'Cross-track '
        ylb = 'Along-track '
    else:
        xlb = 'X '
        ylb = 'Y '

    if iskm:
        xlb = xlb+'[km]'
        ylb = ylb+'[km]'
    else:
        xlb = xlb+'[m]'
        ylb = ylb+'[m]'

    ax.set_xlabel(xlb)
    ax.set_ylabel(ylb)

    ax.set_aspect('equal', 'box')
    if labelcb is None:
        plt.colorbar(im, ax=ax)
    else:
        plt.colorbar(im, ax=ax, label=labelcb)
    return ax


def custom_plots_spec(ax, im, iswnb=1, alongT_isY=1, klim=None, labelcb=None):
    if iswnb:
        xlb = '$k_x$ [rad/m]'
        ylb = '$k_x$ [rad/m]'
    else:
        xlb = '$k_x / 2 \pi$ [km$^{-1}$]'
        ylb = '$k_y / 2 \pi$ [km$^{-1}$]'

    if alongT_isY:
        ax.set_xlabel(ylb)
        ax.set_ylabel(xlb)
    else:
        ax.set_xlabel(xlb)
        ax.set_ylabel(ylb)

    ax.set_aspect('equal', 'box')
    if labelcb is None:
        plt.colorbar(im, ax=ax)
    else:
        plt.colorbar(im, ax=ax, label=labelcb)

    if klim is not None:
        ax.set_xlim((-klim, klim))
        ax.set_ylim((-klim, klim))
    return ax
