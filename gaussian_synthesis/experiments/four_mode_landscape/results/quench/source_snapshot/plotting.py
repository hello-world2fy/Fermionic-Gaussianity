import _bootstrap

# Plotting is an optional, offline stage: no optimization or sampling happens here.
import argparse
import csv
import json
import os
from pathlib import Path
import tempfile

import numpy as np


BLUE = "#2365a8"
ORANGE = "#d77624"
GREEN = "#37865a"
PURPLE = "#8b5da8"
BLACK = "#262626"
MAGIC_PURITY = 116 / 847


def _read_npz(path):
    with np.load(path, allow_pickle=False) as values:
        return {key: values[key] for key in values.files}


def _read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _column(rows, key):
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def _band(ax, x, data, field, *, label, color, linestyle="-", linewidth=1.8):
    y = np.asarray(data[field])
    ax.plot(x, y, label=label, color=color, linestyle=linestyle,
            linewidth=linewidth)
    if field + "_se" in data:
        se = np.asarray(data[field + "_se"])
        ax.fill_between(x, y - se, y + se, color=color, alpha=.15, linewidth=0)


def _style_axis(ax, xlabel=None, ylabel=None):
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.18, linewidth=.6)
    ax.set_axisbelow(True)


def _phase_axis(ax):
    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi],
                  ["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    ax.set_xlabel(r"Interaction phase $\phi=Ut$")
    ax.set_xlim(0, 2 * np.pi)


def _nearest(parameters, targets):
    return [int(np.argmin(np.abs(parameters - target))) for target in targets]


def _save(fig, output, name, plt):
    paths = []
    for extension in ("png", "pdf"):
        relative = Path("figures") / f"{name}.{extension}"
        fig.savefig(output / relative, dpi=200, bbox_inches="tight",
                    facecolor="white")
        paths.append(str(relative))
    plt.close(fig)
    return paths


def _static_overview(x, vectors, obs, summary, plt):
    fig, axes = plt.subplots(2, 2, figsize=(11.3, 8.0), layout="constrained")
    ax = axes[0, 0]
    ax.plot(x, _column(summary, "extent_exact"), color=BLACK,
            label=r"Exact Gaussian extent $\xi_G$", linewidth=2)
    _band(ax, x, obs, "cost_p", label=r"Continuous cost $B_2^2$", color=BLUE)
    ax.set_title("Finite-order cost and exact extent")
    _style_axis(ax, r"Cat parameter $r$", "Coefficient cost")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(x, np.abs(vectors["u"][:, 0]), color=BLUE, label=r"Witness $|a|$")
    ax.plot(x, np.abs(vectors["u"][:, -1]), color=ORANGE, label=r"Witness $|b|$")
    ax.plot(x, np.sqrt(1 - x), color=BLUE, linestyle="--", alpha=.65,
            label=r"Target $\sqrt{1-r}$")
    ax.plot(x, np.sqrt(x), color=ORANGE, linestyle="--", alpha=.65,
            label=r"Target $\sqrt{r}$")
    ax.set_title("Optimization changes relative amplitudes")
    _style_axis(ax, r"Cat parameter $r$", "Amplitude (witness not normalized)")
    ax.legend(ncols=2, fontsize=8.5)

    ax = axes[1, 0]
    _band(ax, x, obs, "purity_p", label=r"Optimized $p_2$", color=BLUE)
    _band(ax, x, obs, "purity_h", label=r"Control $h_2$", color=ORANGE)
    ax.scatter([.5], [MAGIC_PURITY], marker="D", s=34, color=BLACK, zorder=4,
               label=r"Magic reference $116/847$")
    if "purity_cross_p" in obs:
        ax.plot(x, obs["purity_cross_p"], color=BLUE, linestyle=":", alpha=.65,
                label="Independent-block cross estimate")
    ax.set_title("Auxiliary ensemble purity")
    _style_axis(ax, r"Cat parameter $r$", r"$\mathcal{P}_2=\mathrm{Tr}(R_2^2)$")
    ax.legend(fontsize=8.3)

    ax = axes[1, 1]
    _band(ax, x, obs, "tv", label=r"$\frac{1}{2}\int |p_2-h_2|\,d\mu$", color=PURPLE)
    ax.set_title("Distribution difference vanishes at both endpoints")
    _style_axis(ax, r"Cat parameter $r$", "Total variation distance")
    ax.set_ylim(bottom=0)
    ax.legend()
    fig.suptitle("Static four-mode cat: Gaussian configuration landscape", fontsize=15)
    fig.text(.5, -.015, "Bands: +/-1 standard error across independent repeats. "
             "Auxiliary purity is not the purity of the physical wavefunction.",
             ha="center", fontsize=9, color="#555555")
    return fig


def _static_occupancy(x, obs, plt):
    indices = _nearest(x, [0, .1, .25, .5])
    fig, axes = plt.subplots(1, 4, figsize=(13.3, 3.7), layout="constrained",
                             sharex=True, sharey=True)
    edges = obs["n_edges"]
    for ax, i in zip(axes, indices):
        ax.stairs(obs["hist_n_p"][i], edges, color=BLUE, linewidth=1.8,
                  label=r"Optimized $p_2$")
        ax.stairs(obs["hist_n_h"][i], edges, color=ORANGE, linewidth=1.6,
                  linestyle="--", label=r"Control $h_2$")
        ax.set_title(fr"$r={x[i]:.2f}$")
        _style_axis(ax, r"Configuration mean occupation $N(g)$")
        ax.set_xlim(0, 4)
    axes[0].set_ylabel("Probability mass per bin")
    axes[-1].legend(fontsize=8.5)
    fig.suptitle("Static occupancy marginals", fontsize=14)
    fig.text(.5, -.03, "These are distributions over Gaussian configurations, "
             "not particle-number measurement probabilities.", ha="center", fontsize=9)
    return fig


def _configuration_grid(data, row_fields, row_labels, indices, column_labels,
                        title, plt, colors):
    ncols = len(indices)
    fig, axes = plt.subplots(len(row_fields), ncols, figsize=(3 * ncols + .5, 6.1),
                             layout="constrained", sharex=True, sharey=True,
                             squeeze=False)
    selected = np.concatenate([np.asarray(data[field])[indices].reshape(-1)
                               for field in row_fields])
    maximum = max(float(np.max(selected)), 1e-15)
    norm = colors.PowerNorm(gamma=.5, vmin=0, vmax=maximum)
    edges = data["xy_edges"]
    image = None
    for row, (field, label) in enumerate(zip(row_fields, row_labels)):
        for col, (i, column_label) in enumerate(zip(indices, column_labels)):
            ax = axes[row, col]
            image = ax.pcolormesh(edges, edges, data[field][i].T,
                                 cmap="magma", norm=norm, shading="flat", rasterized=True)
            ax.plot([0, 1], [1, 0], color="white", linestyle="--", linewidth=.6,
                    alpha=.5)
            ax.set_aspect("equal")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xticks([0, .5, 1])
            ax.set_yticks([0, .5, 1])
            if row == 0:
                ax.set_title(column_label)
            if col == 0:
                ax.set_ylabel(label + "\n" + r"$|\langle1111|g\rangle|^2$")
            if row == len(row_fields) - 1:
                ax.set_xlabel(r"$|\langle0000|g\rangle|^2$")
    fig.colorbar(image, ax=axes, fraction=.023, pad=.015,
                 label="Probability mass per bin (square-root color scale)")
    fig.suptitle(title, fontsize=14)
    return fig


def _quench_overview(x, obs, reference, control, summary, plt):
    fig, axes = plt.subplots(2, 2, figsize=(11.3, 8.1), layout="constrained")
    ax = axes[0, 0]
    ax.plot(x, _column(summary, "extent_exact"), color=BLACK, linewidth=2,
            label=r"Exact $\xi_G=1+|\sin(\phi/2)|$")
    _band(ax, x, obs, "cost_p", color=BLUE, label=r"Continuous cost $B_2^2$")
    ax.set_title("Interaction creates and removes Gaussian magic")
    _style_axis(ax, ylabel="Coefficient cost")
    ax.legend()

    ax = axes[0, 1]
    _band(ax, x, obs, "purity_p", label="Direct interaction evolution", color=BLUE)
    _band(ax, x, reference, "purity_p", label="Static Schmidt prediction", color=PURPLE,
          linestyle="--")
    _band(ax, x, control, "purity_p", label="Quadratic evolution control", color=GREEN,
          linestyle=":")
    ax.scatter([np.pi], [MAGIC_PURITY], marker="D", s=30, color=BLACK, zorder=4,
               label=r"Magic reference $116/847$")
    ax.set_title("Gaussian-unitary invariant auxiliary purity")
    _style_axis(ax, ylabel=r"$\mathcal{P}_2=\mathrm{Tr}(R_2^2)$")
    ax.legend(fontsize=8.2)

    ax = axes[1, 0]
    _band(ax, x, obs, "tv", label="Direct interaction evolution", color=BLUE)
    _band(ax, x, reference, "tv", label="Static Schmidt prediction", color=PURPLE,
          linestyle="--")
    _band(ax, x, control, "tv", label="Quadratic control", color=GREEN, linestyle=":")
    ax.set_title("Optimized versus direct-overlap distribution")
    _style_axis(ax, ylabel="Total variation distance")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8.3)

    ax = axes[1, 1]
    for field, label, color in (("purity_p", "Purity difference", BLUE),
                                 ("tv", "TV-distance difference", ORANGE)):
        delta = obs[field] - reference[field]
        se = np.sqrt(obs.get(field + "_se", np.zeros_like(x))**2
                     + reference.get(field + "_se", np.zeros_like(x))**2)
        ax.plot(x, delta, color=color, label=label)
        ax.fill_between(x, delta - se, delta + se, color=color, alpha=.15, linewidth=0)
    ax.axhline(0, color=BLACK, linewidth=.8, linestyle="--")
    ax.set_title("Direct result minus static prediction")
    _style_axis(ax, ylabel="Invariant difference")
    ax.legend(fontsize=8.5)
    for ax in axes.flat:
        _phase_axis(ax)
    fig.suptitle("Quartic interaction: direct evolution and static prediction", fontsize=15)
    fig.text(.5, -.015, "Bands: +/-1 standard error. Direct and static predictions use "
             "independent observation streams.", ha="center", fontsize=9, color="#555555")
    return fig


def _quench_occupancy(x, obs, reference, plt):
    indices = _nearest(x, [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    labels = [r"$\phi=0$", r"$\phi=\pi/2$", r"$\phi=\pi$",
              r"$\phi=3\pi/2$", r"$\phi=2\pi$"]
    fig, axes = plt.subplots(2, 5, figsize=(15.2, 6.5), layout="constrained",
                             sharex=True, sharey=True)
    for col, (i, label) in enumerate(zip(indices, labels)):
        for row, (p_key, h_key) in enumerate((("hist_n_p", "hist_n_h"),
                                            ("hist_n_pulled_p", "hist_n_pulled_h"))):
            ax = axes[row, col]
            ax.stairs(obs[p_key][i], obs["n_edges"], color=BLUE, linewidth=1.7,
                      label=r"Optimized $p_2$")
            ax.stairs(obs[h_key][i], obs["n_edges"], color=ORANGE, linestyle="--",
                      linewidth=1.3, label=r"Control $h_2$")
            if row == 1:
                ax.stairs(reference["hist_n_p"][i], reference["n_edges"],
                          color=PURPLE, linestyle=":", linewidth=1.5,
                          label="Static prediction")
            _style_axis(ax)
            ax.set_xlim(0, 4)
            if row == 0:
                ax.set_title(label)
            else:
                ax.set_xlabel(r"$N(U_G^\dagger g)$")
    axes[0, 0].set_ylabel("Laboratory coordinates\nProbability mass per bin")
    axes[1, 0].set_ylabel("Static coordinates\nProbability mass per bin")
    axes[0, -1].legend(fontsize=8)
    axes[1, -1].legend(fontsize=8)
    fig.suptitle("Occupancy distributions depend on the coordinate frame", fontsize=14)
    fig.text(.5, -.015, "Only the distributions transformed back to static coordinates "
             "are compared with the cat family.", ha="center", fontsize=9)
    return fig


def _validation(x, summary, anchors, kind, plt):
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.4), layout="constrained")
    ax = axes[0]
    floor = 1e-16
    ax.semilogy(x, np.maximum(_column(summary, "exact_residual"), floor),
                color=BLUE, label=r"$\|C_2(u)-\psi\|$")
    ax.semilogy(x, np.maximum(_column(summary, "witness_error"), floor),
                color=PURPLE, linestyle="--", label="Distance to analytic witness")
    ax.axhline(1e-10, color=BLACK, linestyle=":", linewidth=1.2,
               label=r"Residual tolerance $10^{-10}$")
    ax.set_title("Full-space deterministic solution")
    _style_axis(ax, ylabel="Absolute error (log scale)")
    ax.legend(fontsize=8.5)
    ax = axes[1]
    if anchors:
        anchor_x = _column(anchors, "parameter")
        ax.plot(anchor_x, _column(anchors, "witness_error"), "o-", color=BLUE,
                label="SGD witness error")
        ax.plot(anchor_x, _column(anchors, "reconstruction_upper_bound"), "s-",
                color=ORANGE, label="Independent reconstruction upper bound")
        ax.plot(anchor_x, _column(anchors, "exact_residual"), "^-", color=PURPLE,
                label="Exact residual of SGD witness")
    else:
        ax.text(.5, .5, "No completed SGD validation points", ha="center", va="center",
                transform=ax.transAxes)
    ax.axhline(.01, color=BLUE, linestyle=":", linewidth=1.2,
               label="Witness tolerance 0.01")
    ax.axhline(.05, color=ORANGE, linestyle=":", linewidth=1.2,
               label="Reconstruction tolerance 0.05")
    anchor_max = max((float(row[key]) for row in anchors for key in
                      ("witness_error", "exact_residual", "reconstruction_upper_bound")),
                     default=.05)
    ax.set_ylim(0, max(.08, 1.6 * anchor_max))
    ax.set_title("Independent Haar-SGD anchor validation")
    _style_axis(ax, ylabel="Absolute error / certified upper bound")
    ax.legend(fontsize=8)
    for ax in axes:
        if kind == "quench":
            _phase_axis(ax)
        else:
            ax.set_xlabel(r"Cat parameter $r$")
    fig.suptitle("Numerical and reconstruction validation", fontsize=14)
    fig.text(.5, -.02, "The reconstruction failure budget is 0.05/9 per anchor across "
             "both experiments; plot error bands elsewhere are descriptive.",
             ha="center", fontsize=9)
    return fig


def plot_results(directory):
    """Render saved experiment outputs; return paths relative to directory.

    Re-rendering replaces only figures. It never reuses, selects, or creates
    numerical candidates and does not change the saved statistical results.
    """
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) /
                                              "four-mode-landscape-matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) /
                                               "four-mode-landscape-cache"))
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import colors
    import matplotlib.pyplot as plt

    output = Path(directory).resolve()
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    kind = config["kind"]
    if kind not in ("static", "quench"):
        raise ValueError(f"Unknown experiment kind: {kind!r}")
    vectors = _read_npz(output / "vectors.npz")
    obs = _read_npz(output / "observations.npz")
    summary = _read_csv(output / "summary.csv")
    anchors = _read_csv(output / "anchors.csv")
    x = vectors["parameters"]
    if len(x) != len(summary):
        raise ValueError("Saved parameter and summary lengths disagree")
    (output / "figures").mkdir(exist_ok=True)
    saved = []
    style = {"font.family": "DejaVu Sans", "font.size": 10,
             "axes.titlesize": 11, "axes.labelsize": 10,
             "legend.frameon": False, "legend.fontsize": 9,
             "axes.linewidth": .8, "savefig.facecolor": "white",
             "pdf.fonttype": 42, "ps.fonttype": 42}
    with plt.rc_context(style):
        if kind == "static":
            saved += _save(_static_overview(x, vectors, obs, summary, plt),
                           output, "static_overview", plt)
            saved += _save(_static_occupancy(x, obs, plt), output, "static_occupancy", plt)
            indices = _nearest(x, [0, .1, .25, .5])
            fig = _configuration_grid(obs, ["hist_xy_p", "hist_xy_h"],
                [r"Optimized $p_2$", r"Control $h_2$"], indices,
                [fr"$r={x[i]:.2f}$" for i in indices],
                "Static vacuum / full-occupation configuration weights", plt, colors)
            saved += _save(fig, output, "static_configurations", plt)
        else:
            reference = _read_npz(output / "reference_observations.npz")
            control = _read_npz(output / "control_observations.npz")
            saved += _save(_quench_overview(x, obs, reference, control, summary, plt),
                           output, "quench_overview", plt)
            saved += _save(_quench_occupancy(x, obs, reference, plt),
                           output, "quench_occupancy", plt)
            indices = _nearest(x, [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
            labels = [r"$\phi=0$", r"$\phi=\pi/2$", r"$\phi=\pi$",
                      r"$\phi=3\pi/2$", r"$\phi=2\pi$"]
            fig = _configuration_grid(obs, ["hist_xy_p", "hist_xy_pulled_p"],
                ["Laboratory frame", "Static frame"], indices, labels,
                r"Optimized $p_2$: configuration weights in two coordinate frames",
                plt, colors)
            saved += _save(fig, output, "quench_configurations", plt)
        saved += _save(_validation(x, summary, anchors, kind, plt),
                       output, "validation", plt)
    return saved


def main():
    parser = argparse.ArgumentParser(description="Render saved four-mode experiment data.")
    parser.add_argument("--input", type=Path, required=True,
                        help="Existing static or quench results directory")
    args = parser.parse_args()
    paths = plot_results(args.input)
    print(json.dumps({"input": str(args.input.resolve()), "figures": paths}, indent=2))


if __name__ == "__main__":
    main()
