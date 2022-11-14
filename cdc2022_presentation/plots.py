"""

"""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np


def init_matplotlib_params(save_not_show_fig, show_latex_fig):
    fontsize = 14
    linewidth = 2.0
    gridlinewidth = 0.7

    # Global changes
    matplotlib.rcParams.update({
            # Fonts
            'font.size': fontsize,
            'axes.titlesize': fontsize,
            'axes.labelsize': fontsize,
            'xtick.labelsize': fontsize,
            'ytick.labelsize': fontsize,
            'legend.fontsize': fontsize,
            'figure.titlesize': fontsize,
            # Line width
            'lines.linewidth': linewidth,
            'grid.linewidth': gridlinewidth
        })

    # Backend if saving
    if save_not_show_fig:
        matplotlib.use("pgf")

    # Font if saving or ploting in tex mode
    if save_not_show_fig or show_latex_fig:
        preamble = "".join([#r"\usepackage{amsmath}",
                            #r"\usepackage{amsfonts}",
                            r"\renewcommand{\vec}[1]{\ensuremath{{\underline{#1}}}}",
                            r"\newcommand{\mat}[1]{{\ensuremath{{\mathbf{#1}}}}}"])
        matplotlib.rcParams.update({
            "pgf.texsystem": "pdflatex",   
            'font.family': 'serif',             # Use serif/main font for text elements
            'text.usetex': True,                # Use inline maths for ticks
            'pgf.rcfonts': False,               # Don't setup fonts from matplotlib rc params
            'text.latex.preamble' : preamble,   # Latex preamble when displaying
            'pgf.preamble': preamble            # Latex preamble when saving (pgf)
        })

    return

# From https://scipython.com/book/chapter-7-matplotlib/examples/bmi-data-with-confidence-ellipses/
def get_cov_ellipse(cov, centre, nstd, **kwargs):
    """
    Return a matplotlib Ellipse patch representing the covariance matrix
    cov centred at centre and scaled by the factor nstd.

    """

    # Find and sort eigenvalues and eigenvectors into descending order
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    # The anti-clockwise angle to rotate our ellipse by 
    vx, vy = eigvecs[:,0][0], eigvecs[:,0][1]
    theta = np.arctan2(vy, vx)

    # Width and height of ellipse to draw
    width, height = 2 * nstd * np.sqrt(eigvals) # eigvals positive because covariance is positive semi definite
    return Ellipse(xy=centre, width=width, height=height, angle=np.degrees(theta), **kwargs)


def plot_state_cov(plotter, state_covariance, state, **kwargs):
    # Centre
    state_2d = np.array([state[0], state[2]])

    # 2d covariance
    state_cov_2d = np.array([[state_covariance[0][0], state_covariance[2][0]],
                             [state_covariance[0][2], state_covariance[2][2]]])
    
    # Create and plot ellipse
    ellipse = get_cov_ellipse(state_cov_2d, state_2d, 2, **kwargs)
    return plotter.add_artist(ellipse)

class FilterAbs:
    def predict(self):
        raise NotImplementedError
    
    def update(self, measurement):
        raise NotImplementedError

class KFilter(FilterAbs):
    def __init__(self, n, m, F, Q, H, R, init_state, init_cov):
        self.n = n
        self.m = m
        self.F = F
        self.Q = Q
        self.H = H
        self.R = R
        self.x = init_state
        self.P = init_cov
        return
    
    def predict(self):
        self.x = self.F@self.x
        self.P = self.Q + (self.F@self.P@self.F.T)
        return self.x, self.P
    
    def update(self, measurement):
        S = (self.H@self.P@self.H.T) + self.R
        invS = np.linalg.inv(S)

        K = self.P@self.H.T@invS

        self.x = self.x + K@(measurement - self.H@self.x)
        self.P = self.P - (K@S@K.T)
        return self.x, self.P


def additional_noises():
    f = plt.figure()
    ax = f.add_subplot(111)

    ax.set_xlabel(r"$\vec{y}_k$")
    ax.set_ylabel(r"$p\left(\vec{y}_k\right)$")

    xs = np.arange(0, 10, 0.1)
    mu = 5
    sd1 = 1
    sd2 = 2

    func1 = [1/(sd1*np.sqrt(2*np.pi))*np.exp(-0.5*((x-mu)/sd1)**2) for x in xs]
    func2 = [1/(sd2*np.sqrt(2*np.pi))*np.exp(-0.5*((x-mu)/sd2)**2) for x in xs]

    ax.fill_between(xs, func1, label="Privileged", alpha=0.5)
    ax.fill_between(xs, func2, label="Unprivileged", alpha=0.5)

    ax.set_xticks([mu])
    ax.set_xticklabels([r"$\mat{H}_k \vec{x}_k$"])

    ax.set_yticks([0, 0.5])

    plt.legend()

    plt.savefig('additional_noises.png')
    plt.show()

    return

def linear_example():
    f = plt.figure()
    ax = f.add_subplot(111)

    ax.set_xlabel(r"Location $x$")
    ax.set_ylabel(r"Location $y$")
    f.suptitle(r"Example: Linear Constant Velocity Model ($n=3$)")

    # State dimension
    n = 4

    # Measurement dimension
    m = 2

    # Process model (q = noise strength, t = timestep)
    q = 0.01
    t = 0.5
    F = np.array([[1, t, 0, 0], 
                  [0, 1, 0, 0], 
                  [0, 0, 1, t], 
                  [0, 0, 0, 1]])

    Q = q*np.array([[t**3/3, t**2/2,      0,      0], 
                    [t**2/2,      t,      0,      0], 
                    [     0,      0, t**3/3, t**2/2],
                    [     0,      0, t**2/2,      t]])

    # Measurement models
    H = np.array([[1, 0, 0, 0], 
                  [0, 0, 1, 0]])

    R = np.array([[5, 2], 
                  [2, 5]])
    
    # Ground truth init
    gt_init_state = np.array([0, 1, 0, 1])

    gts = []
    ms = []
    ms2 = []
    ms3 = []
    x = gt_init_state
    for s in range(50):
        v = np.random.multivariate_normal(np.array([0, 0]), R)
        m = H@x + v
        ms.append(m)

        v = np.random.multivariate_normal(np.array([0, 0]), R)
        m = H@x + v
        ms2.append(m)

        v = np.random.multivariate_normal(np.array([0, 0]), R)
        m = H@x + v
        ms3.append(m)

        gts.append(x)

        w = np.random.multivariate_normal(np.array([0, 0, 0, 0]), Q)
        x = F@x + w

    ax.plot([gt[0] for gt in gts], [gt[2] for gt in gts], marker='.', label="Path")
    ax.scatter([m[0] for m in ms], [m[1] for m in ms], marker='x', label="Sensor 1", color='r')
    ax.scatter([m[0] for m in ms2], [m[1] for m in ms2], marker='x', label="Sensor 2", color='g')
    ax.scatter([m[0] for m in ms3], [m[1] for m in ms3], marker='x', label="Sensor 3", color='b')
    ax.scatter(gt_init_state[0], gt_init_state[2], label="Initial Position", zorder=10, color='orange')

    ax.set_xticks([])
    ax.set_yticks([])

    plt.legend()

    plt.savefig('linear_example.png')
    plt.show()

    return

def estimation_privilege():
    f = plt.figure(figsize=(9, 3))
    ax = f.add_subplot(111)

    ax.set_xlabel(r"Location $x$")
    ax.set_ylabel(r"Location $y$")

    # State dimension
    n = 4

    # Measurement dimension
    m = 2

    # Process model (q = noise strength, t = timestep)
    q = 0.01
    t = 0.5
    F = np.array([[1, t, 0, 0], 
                  [0, 1, 0, 0], 
                  [0, 0, 1, t], 
                  [0, 0, 0, 1]])

    Q = q*np.array([[t**3/3, t**2/2,      0,      0], 
                    [t**2/2,      t,      0,      0], 
                    [     0,      0, t**3/3, t**2/2],
                    [     0,      0, t**2/2,      t]])

    # Measurement models
    H = np.array([[1, 0, 0, 0], 
                  [0, 0, 1, 0]])

    R = np.array([[5, 2], 
                  [2, 5]])

    Z = np.array([[10, 0],
                  [0, 10]])

    # Filter init
    init_state = np.array([0, 1, 0, 0.1])
    init_cov = np.array([[0, 0, 0, 0], 
                         [0, 0, 0, 0], 
                         [0, 0, 0, 0], 
                         [0, 0, 0, 0]])
    
    # Ground truth init
    gt_init_state = np.array([0, 1, 0, 0.1])

    priv_kf = KFilter(n, m, F, Q, H, R, init_state, init_cov)
    unpriv_kf = KFilter(n, m, F, Q, H, R+Z, init_state, init_cov)

    gts = []
    privs = [(priv_kf.x, priv_kf.P)]
    unprivs = [(unpriv_kf.x, unpriv_kf.P)]
    x = gt_init_state
    sim_steps = 51
    for s in range(sim_steps):
        v1 = np.random.multivariate_normal(np.array([0, 0]), R)
        v2 = np.random.multivariate_normal(np.array([0, 0]), R+Z)
        m1 = H@x + v1
        m2 = H@x + v2

        priv_kf.predict()
        unpriv_kf.predict()
        
        gts.append(x)
        privs.append(priv_kf.update(m1))
        unprivs.append(unpriv_kf.update(m2))

        w = np.random.multivariate_normal(np.array([0, 0, 0, 0]), Q)
        x = F@x + w

    start_skip = 8

    ax.plot([gt[0] for gt in gts[start_skip:]], [gt[2] for gt in gts[start_skip:]], marker='.', label="Path", color='lightgrey')
    ax.plot([p[0][0] for p in privs[start_skip:]], [p[0][2] for p in privs[start_skip:]], marker='.', label="Privileged", color='green')
    ax.plot([p[0][0] for p in unprivs[start_skip:]], [p[0][2] for p in unprivs[start_skip:]], marker='.', label="Unprivileged", color='darkred')

    for s in range(start_skip, sim_steps):
        if s % 5 == 0:
            plot_state_cov(ax, unprivs[s][1], unprivs[s][0], fill=True, linestyle='', color='red', alpha=0.2)

    for s in range(start_skip, sim_steps):
        if s % 5 == 0:
            plot_state_cov(ax, privs[s][1], privs[s][0], fill=True, linestyle='', color='green', alpha=0.2)
            

    #ax.scatter(gt_init_state[0], gt_init_state[2], label="Initial Estimate", zorder=10, color='orange')

    ax.set_xticks([])
    ax.set_yticks([])

    plt.legend()

    plt.savefig('estimation_privilege.png')
    plt.show()
    return


init_matplotlib_params(False, True)
#additional_noises()
linear_example()
#estimation_privilege()