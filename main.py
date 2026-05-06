import numpy as np
import matplotlib.pyplot as plt
 
# ── settings SI UNIT ──────────────────────────────────────────────
g   = 9.81   # gravity 
dx  = 2.0    # horizontal distance 
dy  = 1.5    # vertical drop 
mu  = 0.0    # friction coefficient 
N   = 500    # number of points along each curve

# ── 1. define the four curves ──────────────────────────────
# each curve returns arrays x[], y[] going from (0,0) to (dx, dy)

t = np.linspace(0, 1, N)

# straight line
x_line = t * dx
y_line = t * dy

# parabola
x_para = t * dx
y_para = t * dy + np.sin(np.pi * t) * dy * 0.5

# circular arc
angle_start = np.arctan2(0 - dy, 0 - 0)       # angle from circle centre to start
angle_end   = np.arctan2(dy - dy, dx - 0)      # angle from circle centre to end
cx, cy      = 0, dy                             # circle centre
radius      = np.sqrt(dx**2 + dy**2)
angles      = np.linspace(angle_start, angle_end, N)
x_circ      = cx + radius * np.cos(angles)
y_circ      = cy + radius * np.sin(angles)

# cycloid — the theoretical winner
# parametric form: x = r(t - sin t), y = r(1 - cos t)
#  solve for the parameter T that makes the curve hit (dx, dy)
def find_T(dx, dy, T_guess=2.0):
    ratio = dy / dx
    T = T_guess
    for _ in range(100):                         # Newton METHOD
        fx  = (1 - np.cos(T)) / (T - np.sin(T)) - ratio
        dfx = (np.sin(T) * (T - np.sin(T)) - (1 - np.cos(T))**2) / (T - np.sin(T))**2
        T  -= fx / dfx
        if abs(fx) < 1e-12:
            break
    return T

T_end  = find_T(dx, dy)
r      = dx / (T_end - np.sin(T_end))
tp     = np.linspace(0, T_end, N)
x_cycl = r * (tp - np.sin(tp))
y_cycl = r * (1 - np.cos(tp))


# ── 2. simulate bead travel time ──────────────────────────
# v² = v₀² + 2*(g_tangential - friction)*ds

def travel_time(xs, ys, mu):
    v   = 0.0          # starts from rest
    t   = 0.0
    for i in range(1, len(xs)):
        ddx = xs[i] - xs[i-1]
        ddy = ys[i] - ys[i-1]
        ds  = np.sqrt(ddx**2 + ddy**2)
        if ds < 1e-12:
            continue
        # angle of slope (positive = going downhill)
        angle      = np.arctan2(ddy, ddx)
        g_along    = g * np.sin(angle)              # gravity component along path
        g_perp     = g * np.cos(angle)              # normal force component
        friction   = mu * g_perp                    # friction force per unit mass
        net_accel  = g_along - friction
        v2_new     = v**2 + 2 * net_accel * ds
        if v2_new > 0:
            v_new  = np.sqrt(v2_new)
        else:
            v_new  = 1e-6                           # nearly stopped
        v_avg  = (v + v_new) / 2 if v + v_new > 0 else 1e-6
        t     += ds / v_avg
        v      = v_new
    return t


# ── 3. compute times for all curves ───────────────────────
curves = {
    'Straight line'  : (x_line, y_line),
    'Parabola'       : (x_para, y_para),
    'Circular arc'   : (x_circ, y_circ),
    'Cycloid'        : (x_cycl, y_cycl),
}

print(f"Friction μ = {mu}\n{'─'*30}")
times = {}
for name, (xs, ys) in curves.items():
    times[name] = travel_time(xs, ys, mu)
    print(f"{name:<18}: {times[name]:.4f} s")

fastest = min(times, key=times.get)
print(f"\nFastest: {fastest}")


# ── 4. plot the curves ─────────────────────────────────────
colors = {
    'Straight line' : '#888',
    'Parabola'      : '#185FA5',
    'Circular arc'  : '#3B6D11',
    'Cycloid'       : '#993C1D',
}
styles = {
    'Straight line' : '--',
    'Parabola'      : '-.',
    'Circular arc'  : ':',
    'Cycloid'       : '-',
}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# left plot: the curves
ax = axes[0]
ax.set_title('The four curves')
for name, (xs, ys) in curves.items():
    t_label = f'{name}  ({times[name]:.3f} s)'
    ax.plot(xs, ys, linestyle=styles[name], color=colors[name],
            linewidth=2 if name == 'Cycloid' else 1.4, label=t_label)
ax.plot(0, 0, 'ko', ms=7)
ax.plot(dx, dy, 'ko', ms=7)
ax.set_xlabel('Horizontal distance (m)')
ax.set_ylabel('Vertical drop (m)')
ax.invert_yaxis()
ax.set_aspect('equal')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# right plot: time vs friction coefficient
ax2 = axes[1]
ax2.set_title('Travel time vs friction (your original result)')
mu_values = np.linspace(0, 0.5, 40)
for name, (xs, ys) in curves.items():
    ts = [travel_time(xs, ys, m) for m in mu_values]
    ax2.plot(mu_values, ts, linestyle=styles[name], color=colors[name],
             linewidth=2 if name == 'Cycloid' else 1.4, label=name)
ax2.set_xlabel('Friction coefficient μ')
ax2.set_ylabel('Travel time (s)')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('brachistochrone.png', dpi=150)
plt.show()
print('\nPlot saved as brachistochrone.png')
