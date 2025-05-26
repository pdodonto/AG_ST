import matplotlib.pyplot as plt
import io

def plot_spectrum(dfs, colors, labels):
    fig, ax = plt.subplots()
    for df, color, label in zip(dfs, colors, labels):
        ax.plot(df.values[0], color=color, label=label)
    ax.legend()

    # Salvar em memória
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf