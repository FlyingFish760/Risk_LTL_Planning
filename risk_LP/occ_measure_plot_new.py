import numpy as np
import matplotlib.pyplot as plt

def plot_occ_measure(
    occ_measure, prod_auto, abs_model,
    cmap_main="YlOrRd", cmap_vel="YlOrRd",
    annotate=True, show_grid=True,
    dpi=300, save_path=None
):
    """
    Plot occupation measures with thesis-friendly styling.

    Args:
        occ_measure: dict[((state_ind, action_ind)) -> float]
        prod_auto: object with prod_state_set (indexable)
        abs_model: object with state_set, action_set (indexable)
        cmap_main: colormap for positional heatmap
        cmap_vel: colormap for velocity subplots
        annotate: whether to draw numeric annotations
        show_grid: whether to add light gridlines around cells
        dpi: figure DPI for saving
        save_path: if not None, path (without extension or with .png/.pdf)
                   - main figure: "<save_path>_pos.png/pdf"
                   - velocity figure: "<save_path>_vel.png/pdf"
    """

    # -----------------------------
    # Common helpers and styling
    # -----------------------------
    def _add_cell_grid(ax, r_max, ey_max):
        if not show_grid:
            return
        ax.set_xticks(np.arange(-.5, r_max, 1), minor=True)
        ax.set_yticks(np.arange(-.5, ey_max, 1), minor=True)
        ax.grid(which="minor", color="gray", linestyle="--", linewidth=0.5, alpha=0.4)
        ax.tick_params(which="minor", bottom=False, left=False)

    def _set_axes_labels(ax, title):
        ax.set_xlabel('r (longitudinal position)')
        ax.set_ylabel('ey (lateral position)')
        ax.set_title(title)

    # Make text and backgrounds clean for print
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.titlesize": 14,
    })

    # ============================================================
    # 1) Positional occupation measure (aggregated over velocities)
    # ============================================================
    state_action_occ = {}

    for (state_ind, action_ind), occ_value in occ_measure.items():
        x, s_cs, s_s = prod_auto.prod_state_set[state_ind]
        (r, ey, v) = abs_model.state_set[x]
        (m, s) = abs_model.action_set[action_ind]

        if (r, ey) not in state_action_occ:
            state_action_occ[(r, ey)] = {}
        if m not in state_action_occ[(r, ey)]:
            state_action_occ[(r, ey)][m] = 0.0
        state_action_occ[(r, ey)][m] += float(occ_value)

    state_occ = {}
    for (r, ey), m_occ in state_action_occ.items():
        state_occ[(r, ey)] = state_occ.get((r, ey), 0.0) + sum(m_occ.values())

    if state_occ:
        r_max = max(r for (r, ey) in state_occ.keys()) + 1
        ey_max = max(ey for (r, ey) in state_occ.keys()) + 1
    else:
        # Nothing to plot
        r_max = ey_max = 1

    heat_map = np.zeros((ey_max, r_max))
    for (r, ey), occ_value in state_occ.items():
        heat_map[ey, r] = occ_value

    fig = plt.figure(figsize=(12, 8), dpi=dpi)
    ax = plt.gca()
    im = ax.imshow(heat_map, cmap=cmap_main, interpolation='nearest', origin='lower')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Occupation Measure')

    _add_cell_grid(ax, r_max, ey_max)

    # numeric annotations at states
    if annotate:
        for (r, ey), occ_value in state_occ.items():
            ax.text(
                r, ey - 0.3, f'{occ_value:.3f}',
                ha='center', va='center',
                color='black', fontsize=9, weight='bold'
            )

    # Action arrows/labels
    arrow_scale = 0.15
    for (r, ey), action_occ in state_action_occ.items():
        start_x, start_y = r, ey + 0.2
        max_action_type = max(action_occ.items(), key=lambda x: x[1])[0] if action_occ else None

        for action_type, action_value in action_occ.items():
            dx, dy = 0.0, 0.0
            action_label = ""
            if action_type == 'l':   # left+forward (your semantics)
                dx, dy = arrow_scale, arrow_scale
                action_label = "L"
            elif action_type == 'r': # right+forward
                dx, dy = arrow_scale, -arrow_scale
                action_label = "R"
            elif action_type == 'f': # forward
                dx, dy = arrow_scale, 0.0
                action_label = "F"

            if dx != 0.0 or dy != 0.0:
                arrow_color = 'darkgreen' if action_type == max_action_type else 'darkred'
                ax.arrow(
                    start_x, start_y, dx, dy,
                    head_width=0.06, head_length=0.03,
                    fc=arrow_color, ec=arrow_color, alpha=0.85, linewidth=1.2, length_includes_head=True
                )
                ax.text(
                    start_x + dx + 0.05, start_y + dy,
                    f'{action_label}: {action_value:.2f}',
                    ha='left', va='center', color=arrow_color, fontsize=7, weight='bold'
                )

    _set_axes_labels(ax, 'State Occupation Measure with Action Directions')
    fig.tight_layout()

    if save_path:
        # choose extension: if user provides one, use same; otherwise default png
        if save_path.lower().endswith(('.png', '.pdf', '.svg', '.eps', '.jpg', '.jpeg', '.tif', '.tiff')):
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        else:
            fig.savefig(f"{save_path}_pos.png", dpi=dpi, bbox_inches='tight')

    plt.show()

    # ==========================================
    # 2) Velocity-conditioned occupation measure
    # ==========================================
    velocity_values = set()
    for (state_ind, action_ind), occ_value in occ_measure.items():
        x, s_cs, s_s = prod_auto.prod_state_set[state_ind]
        (r, ey, v) = abs_model.state_set[x]
        velocity_values.add(v)

    velocity_values = sorted(list(velocity_values))
    n_velocities = len(velocity_values)

    if n_velocities == 0:
        return

    n_cols = min(3, n_velocities)
    n_rows = (n_velocities + n_cols - 1) // n_cols

    fig2, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 6*n_rows), dpi=dpi)
    # Normalize axes to 2D array for easy indexing
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = np.array([[ax] for ax in axes])

    for idx, v_val in enumerate(velocity_values):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        state_action_occ_velocity = {}
        for (state_ind, action_ind), occ_value in occ_measure.items():
            x, s_cs, s_s = prod_auto.prod_state_set[state_ind]
            (r, ey, v) = abs_model.state_set[x]
            (m, s) = abs_model.action_set[action_ind]
            if v != v_val:
                continue
            if (r, ey) not in state_action_occ_velocity:
                state_action_occ_velocity[(r, ey)] = {}
            if s not in state_action_occ_velocity[(r, ey)]:
                state_action_occ_velocity[(r, ey)][s] = 0.0
            state_action_occ_velocity[(r, ey)][s] += float(occ_value)

        state_occ_velocity = {k: sum(d.values()) for k, d in state_action_occ_velocity.items()}

        if not state_occ_velocity:
            ax.set_title(f'Velocity {v_val} (No Data)')
            ax.axis('off')
            continue

        r_max_v = max(r for (r, ey) in state_occ_velocity.keys()) + 1
        ey_max_v = max(ey for (r, ey) in state_occ_velocity.keys()) + 1

        heat_map_velocity = np.zeros((ey_max_v, r_max_v))
        for (r, ey), occ_value in state_occ_velocity.items():
            heat_map_velocity[ey, r] = occ_value

        im = ax.imshow(heat_map_velocity, cmap=cmap_vel, interpolation='nearest', origin='lower')

        if annotate:
            for (r, ey), occ_value in state_occ_velocity.items():
                ax.text(
                    r, ey - 0.3, f'{occ_value:.3f}',
                    ha='center', va='center',
                    color='black', fontsize=8, weight='bold'
                )

        # Speed action labels at each state
        # 'a' accelerate, 'd' decelerate, 'c' cruise
        label_positions = {'a': (0.22, 0.22), 'd': (-0.22, 0.22), 'c': (0.0, 0.33)}
        for (r, ey), action_occ in state_action_occ_velocity.items():
            max_action_type = max(action_occ.items(), key=lambda x: x[1])[0] if action_occ else None
            for action_type, action_value in action_occ.items():
                action_label = {'a': 'A', 'd': 'D', 'c': 'C'}.get(action_type, '?')
                text_color = 'darkgreen' if action_type == max_action_type else 'darkred'
                dx, dy = label_positions.get(action_type, (0.0, 0.0))
                ax.text(
                    r + dx, ey + dy, f'{action_label}: {action_value:.2f}',
                    ha='center', va='center', color=text_color, fontsize=7, weight='bold'
                )

        _add_cell_grid(ax, r_max_v, ey_max_v)
        _set_axes_labels(ax, f'Speed Actions at Velocity {v_val}')

    # Hide unused axes (if any)
    total_axes = n_rows * n_cols
    for idx in range(n_velocities, total_axes):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')

    # Add a single colorbar for the velocity figure (optional but useful)
    # We create a shared mappable using the last im if present
    fig2.tight_layout()
    if save_path:
        if save_path.lower().endswith(('.png', '.pdf', '.svg', '.eps', '.jpg', '.jpeg', '.tif', '.tiff')):
            out_path = save_path
            # avoid overwriting positional figure if same name; append suffix
            dot = out_path.rfind(".")
            fig2.savefig(out_path[:dot] + "_vel" + out_path[dot:], dpi=dpi, bbox_inches='tight')
        else:
            fig2.savefig(f"{save_path}_vel.png", dpi=dpi, bbox_inches='tight')

    plt.show()
