from scipy.integrate import odeint
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse, Rectangle

from pydrake.multibody.rigid_body_tree import  RigidBodyTree
from pydrake.multibody.rigid_body_tree import FloatingBaseType
from pydrake.trajectories import PiecewisePolynomial

from underactuated import (FindResource, PlanarRigidBodyVisualizer)

def simWithoutStabilization(plant, state_initial, time_array, uncertainty=False):
    # take input trajectory, do better integration of dynamics

    # wrapper around dynamics
    def dxdt(x, t):
        dynamics = plant.airplaneLongDynamics(x, plant.udtraj_poly.value(t), uncertainty=uncertainty)
        return dynamics

    states_over_time = odeint(dxdt, state_initial, time_array)
    return states_over_time


def simTvlqrStabilization(plant, state_initial, time_array, uncertainty=False):
    # take input trajectory, do first-order integration of dynamics with TVLQR stabilization

    u_over_time = np.zeros((len(time_array)-1, plant.num_inputs))

    if plant.K is None:
        print 'computing TVLQR along trajectory'
        K = plant.calcTvlqrStabilization()

    # wrapper around dynamics with TVLQR feedback
    def dxdt(x, t):
        u0 = plant.udtraj_poly.value(t)
        x0 = plant.xdtraj_poly.value(t)

        ud = plant.Klookup(t).dot(x.flatten() - x0.flatten())
        u = u0.flatten() - ud.flatten()
        dynamics = plant.airplaneLongDynamics(x, u, uncertainty=uncertainty)
        return dynamics

    states_over_time = odeint(dxdt, state_initial, time_array)

    # reconstruct input using states_over_time vs x0
    for i, t in enumerate(time_array[:-1]):
        u0 = plant.udtraj_poly.value(t)
        x0 = plant.xdtraj_poly.value(t)
        u_over_time[i] = u0.flatten() - plant.Klookup(t).dot(states_over_time[i].flatten() - x0.flatten()).flatten()

    return states_over_time, u_over_time


def simAndCompare(plant, x0, obs=None, goalnode=None, compare=True):
    # simulate dynamics at finer time discretization
    input_lookup = np.vectorize(plant.udtraj_poly.value)

    times_fine = np.linspace(0.0, plant.ttraj[-1], 20*len(plant.ttraj))
    utraj_fine = input_lookup(times_fine[0])
    for t in times_fine[1:]:
        utraj_fine = np.hstack((utraj_fine, input_lookup(t)))
    utraj_fine = utraj_fine.T

    xtraj_fine_nostab_nouncert = simWithoutStabilization(plant, x0, times_fine, uncertainty=False)
    xtraj_fine_nostab = simWithoutStabilization(plant, x0, times_fine, uncertainty=True)
    xtraj_fine, utraj_fine_stab = simTvlqrStabilization(plant, x0, times_fine, uncertainty=True)
    print "finer dynamics simulated"

    if compare:
        fig, ax = plt.subplots(figsize=(12, 2.5))
        plt.plot(plant.xdtraj[:, 0] - plant.xdtraj[:, 0].max(), plant.xdtraj[:, 1], 'o', mfc='orange')
        plt.plot(
                 xtraj_fine_nostab_nouncert[:, 0] - plant.xdtraj[:, 0].max(), xtraj_fine_nostab_nouncert[:, 1], 'k',
                 xtraj_fine_nostab[:, 0] - plant.xdtraj[:, 0].max(), xtraj_fine_nostab[:, 1], 'g',
                 xtraj_fine[:, 0] - plant.xdtraj[:, 0].max(), xtraj_fine[:, 1], 'b', linewidth=2)
        plt.title('trajectory in x-z plane')
        plt.xlabel('x [m]')
        plt.ylabel('z [m]')
        plt.xlim((plant.xdtraj[:, 0].min() - 1 - plant.xdtraj[:, 0].max(), plant.xdtraj[:, 0].max() + 1 - plant.xdtraj[:, 0].max()))
        plt.ylim((0, 2))
        if obs is not None:
            for ob in obs:
                ax.add_patch(Rectangle((ob[0], ob[1]), ob[2], ob[3], color='orange', alpha=0.5))
            plt.legend(('(piecewise) optimal', 'unstabilized (no uncert.)', 'unstabilized', 'stabilized'), loc='best')
            # plt.legend(('(piecewise) optimal', 'stabilized'), loc='best')
        else:
            plt.legend(('optimal', 'unstabilized (no uncert.)', 'unstabilized', 'stabilized'), loc='best')
        if goalnode is not None:
            plt.plot(plant.xdtraj[-1, 0], plant.xdtraj[-1, 1], 'gp', ms=16, alpha=0.5)
            goalnode = goalnode.parent
            while goalnode.parent is not None:
                plt.plot(goalnode.state[0], goalnode.state[1], 'gp', ms=16, alpha=0.5)
                goalnode = goalnode.parent
        plt.tight_layout()
        # plt.savefig('figs/stab_traj_fig1.png', dpi=300)
        plt.show()

        plt.figure(figsize=(10, 7))
        plt.subplot(321)
        plt.plot(plant.ttraj, plant.xdtraj[:,0] - plant.xdtraj[:,0].max(), 'o', mfc='orange')
        plt.plot(times_fine, xtraj_fine_nostab_nouncert[:,0] - plant.xdtraj[:,0].max(), 'k',
                 times_fine, xtraj_fine_nostab[:,0] - plant.xdtraj[:,0].max(), 'g',
                 times_fine, xtraj_fine[:,0] - plant.xdtraj[:,0].max(), 'b')
        plt.title('x-position')
        plt.ylabel('x [m]')

        plt.subplot(322)
        plt.plot(plant.ttraj, plant.xdtraj[:,1], 'o', mfc='orange')
        plt.plot(times_fine, xtraj_fine_nostab_nouncert[:,1], 'k',
                 times_fine, xtraj_fine_nostab[:,1], 'g',
                 times_fine, xtraj_fine[:,1], 'b')
        plt.title('z-position')
        plt.ylabel('z [m]')

        plt.subplot(323)
        plt.plot(plant.ttraj, plant.xdtraj[:,2], 'o', mfc='orange')
        plt.plot(times_fine, xtraj_fine_nostab_nouncert[:,2], 'k',
                 times_fine, xtraj_fine_nostab[:,2], 'g',
                 times_fine, xtraj_fine[:,2], 'b')
        plt.title('velocity')
        plt.ylabel('V [m/s]')

        plt.subplot(324)
        plt.plot(plant.ttraj, plant.xdtraj[:,3]*180/np.pi, 'o', mfc='orange')
        plt.plot(times_fine, xtraj_fine_nostab_nouncert[:,3]*180/np.pi, 'k',
                 times_fine, xtraj_fine_nostab[:,3]*180/np.pi, 'g',
                 times_fine, xtraj_fine[:,3]*180/np.pi, 'b')
        plt.title('flight path angle')
        plt.ylabel('gamma [deg]')

        plt.subplot(325)
        plt.plot(plant.ttraj, plant.xdtraj[:,4]*180/np.pi, 'o', mfc='orange')
        plt.plot(times_fine, xtraj_fine_nostab_nouncert[:,4]*180/np.pi, 'k',
                 times_fine, xtraj_fine_nostab[:,4]*180/np.pi, 'g',
                  times_fine, xtraj_fine[:,4]*180/np.pi, 'b')
        plt.title('pitch angle')
        plt.ylabel('theta [deg]')

        plt.subplot(326)
        plt.plot(plant.ttraj, plant.xdtraj[:,5]*180/np.pi, 'o', mfc='orange')
        plt.plot(times_fine, xtraj_fine_nostab_nouncert[:,5]*180/np.pi, 'k',
                 times_fine, xtraj_fine_nostab[:,5]*180/np.pi, 'g',
                 times_fine, xtraj_fine[:,5]*180/np.pi, 'b')
        plt.title('pitch rate')
        plt.ylabel('q [deg/s]')

        plt.tight_layout()
        # plt.savefig('figs/stab_state_fig1.png', dpi=300)
        plt.show()

        # NOTE this version doesn't have right slice for dircol
        plt.figure(figsize=(10,2.5))
        plt.subplot(121)
        plt.plot(plant.ttraj[0:-1], plant.udtraj[:,0], 'o', mfc='orange')
        plt.plot(times_fine, utraj_fine[:,0], 'g',
                 times_fine[0:-1], utraj_fine_stab[:,0], 'b')
        plt.title('thrust')
        plt.ylabel('thrust [N]')

        plt.subplot(122)
        plt.plot(plant.ttraj[0:-1], plant.udtraj[:,1], 'o', mfc='orange')
        plt.plot(times_fine, utraj_fine[:,1], 'g',
                 times_fine[0:-1], utraj_fine_stab[:,1], 'b')
        plt.title('elevator')
        plt.ylabel('elevator [deg]')
        plt.tight_layout()
        # plt.savefig('figs/stab_inp_fig1.png', dpi=300)

    else:
        fig, ax = plt.subplots(figsize=(12, 2.5))
        plt.plot(plant.xdtraj[:, 0] - plant.xdtraj[:, 0].max(), plant.xdtraj[:, 1], 'ro',
                 xtraj_fine_nostab_nouncert[:, 0] - plant.xdtraj[:, 0].max(), xtraj_fine_nostab_nouncert[:, 1], 'k',
                 xtraj_fine_nostab[:, 0] - plant.xdtraj[:, 0].max(), xtraj_fine_nostab[:, 1], 'g')
        plt.title('trajectory in x-z plane')
        plt.xlabel('x [m]')
        plt.ylabel('z [m]')
        plt.xlim((plant.xdtraj[:, 0].min() - 1 - plant.xdtraj[:, 0].max(), plant.xdtraj[:, 0].max() + 1 - plant.xdtraj[:, 0].max()))
        plt.ylim((0, 2))
        if obs is not None:
            for ob in obs:
                ax.add_patch(Rectangle((ob[0], ob[1]), ob[2], ob[3], color='orange', alpha=0.5))
            plt.legend(('(piecewise) optimal', 'unstabilized (no uncert.)', 'unstabilized'), loc='best')
        else:
            plt.legend(('optimal', 'unstabilized (no uncert.)', 'unstabilized'), loc='best')
        if goalnode is not None:
            plt.plot(plant.xdtraj[-1, 0], plant.xdtraj[-1, 1], 'gp', ms=16)
            goalnode = goalnode.parent
            while goalnode.parent is not None:
                plt.plot(goalnode.state[0], goalnode.state[1], 'gp', ms=16)
                goalnode = goalnode.parent
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 7))
        plt.subplot(321)
        plt.plot(plant.ttraj, plant.xdtraj[:, 0] - plant.xdtraj[:, 0].max(), 'ro',
                 times_fine, xtraj_fine_nostab_nouncert[:, 0] - plant.xdtraj[:, 0].max(), 'k',
                 times_fine, xtraj_fine_nostab[:, 0] - plant.xdtraj[:, 0].max(), 'g')
        plt.title('x-position')
        plt.ylabel('x [m]')

        plt.subplot(322)
        plt.plot(plant.ttraj, plant.xdtraj[:, 1], 'ro',
                 times_fine, xtraj_fine_nostab_nouncert[:, 1], 'k',
                 times_fine, xtraj_fine_nostab[:, 1], 'g')
        plt.title('z-position')
        plt.ylabel('z [m]')

        plt.subplot(323)
        plt.plot(plant.ttraj, plant.xdtraj[:, 2], 'ro',
                 times_fine, xtraj_fine_nostab_nouncert[:, 2], 'k',
                 times_fine, xtraj_fine_nostab[:, 2], 'g')
        plt.title('velocity')
        plt.ylabel('V [m/s]')

        plt.subplot(324)
        plt.plot(plant.ttraj, plant.xdtraj[:, 3] * 180 / np.pi, 'ro',
                 times_fine, xtraj_fine_nostab_nouncert[:, 3] * 180 / np.pi, 'k',
                 times_fine, xtraj_fine_nostab[:, 3] * 180 / np.pi, 'g')
        plt.title('flight path angle')
        plt.ylabel('gamma [deg]')

        plt.subplot(325)
        plt.plot(plant.ttraj, plant.xdtraj[:, 4] * 180 / np.pi, 'ro',
                 times_fine, xtraj_fine_nostab_nouncert[:, 4] * 180 / np.pi, 'k',
                 times_fine, xtraj_fine_nostab[:, 4] * 180 / np.pi, 'g')
        plt.title('pitch angle')
        plt.ylabel('theta [deg]')

        plt.subplot(326)
        plt.plot(plant.ttraj, plant.xdtraj[:, 5] * 180 / np.pi, 'ro',
                 times_fine, xtraj_fine_nostab_nouncert[:, 5] * 180 / np.pi, 'k',
                 times_fine, xtraj_fine_nostab[:, 5] * 180 / np.pi, 'g')
        plt.title('pitch rate')
        plt.ylabel('q [deg/s]')

        plt.tight_layout()
        plt.show()

        # NOTE this version doesn't have right slice for dircol
        plt.figure(figsize=(12, 2.5))
        plt.subplot(121)
        plt.plot(plant.ttraj[0:-1], plant.udtraj[:, 0], 'ro',
                 times_fine, utraj_fine[:, 0], 'g')
        plt.title('thrust')
        plt.ylabel('thrust [N]')

        plt.subplot(122)
        plt.plot(plant.ttraj[0:-1], plant.udtraj[:, 1], 'ro',
                 times_fine, utraj_fine[:, 1], 'g')
        plt.title('elevator')
        plt.ylabel('elevator [deg]')
        plt.tight_layout()
    return plt


def plotTrajFunnel(plant, rho, knots):
    x0 = plant.xdtraj

    fig = plt.figure(figsize=(12, 3))
    ax = fig.add_subplot(1, 1, 1)

    # plt.plot(x0[knots, 0], x0[knots, 1], c='r')
    plt.plot(x0[:, 0], x0[:, 1], c='r', linewidth=2)

    rz = np.zeros_like(rho)

    for i in range(len(rho)):
        knot = knots[i]

        S = plant.S_lqr[knot]
        S_sub = np.reshape(S[[0, 1, 0, 1], [0, 0, 1, 1]], (2, 2))

        S_sub_z = S[1, 1]
        rz[i] = np.sqrt(rho[i] / np.sqrt(S_sub_z))

        ellipseInfo = np.linalg.eig(S_sub)

        axis_1 = ellipseInfo[1][0, :]
        if ellipseInfo[0][0] > 0 and ellipseInfo[0][1] > 0:
            r1 = np.sqrt(rho[i]) / np.sqrt(ellipseInfo[0][0])
            axis_2 = ellipseInfo[1][1, :]
            r2 = np.sqrt(rho[i]) / np.sqrt(ellipseInfo[0][1])
            # print "Area of your region of attraction: ", np.pi * r1 * r2
            angle = np.arctan2(-axis_1[1], axis_1[0])

            # ax.add_patch(Ellipse((x0[knot, 0], x0[knot, 1]),
            #                      2 * r1, 2 * r2,
            #                      angle=angle * 180. / np.pi,
            #                      linewidth=2, fill=True, alpha=0.2, zorder=2))

    ax.fill_between(x0[knots, 0], x0[knots, 1] - rz, x0[knots, 1] + rz, color='g', alpha=0.2)

    input_lookup = np.vectorize(plant.udtraj_poly.value)

    times_fine = np.linspace(0.0, plant.ttraj[-1], 20*len(plant.ttraj))
    utraj_fine = input_lookup(times_fine[0])
    for t in times_fine[1:]:
        utraj_fine = np.hstack((utraj_fine, input_lookup(t)))
    xtraj_fine_nostab_nouncert = simWithoutStabilization(plant, x0[0,:], times_fine, uncertainty=False)
    xtraj_fine_nostab = simWithoutStabilization(plant, x0[0,:], times_fine, uncertainty=True)
    xtraj_fine, utraj_fine_stab = simTvlqrStabilization(plant, x0[0,:], times_fine, uncertainty=True)
    plt.plot(xtraj_fine[:, 0], xtraj_fine[:, 1], c='b', linewidth=2)
    plt.plot(xtraj_fine_nostab[:, 0], xtraj_fine_nostab[:, 1], c='k', linewidth=2)
    plt.plot(xtraj_fine_nostab_nouncert[:, 0], xtraj_fine_nostab_nouncert[:, 1], c='g', linewidth=2)

    plt.xlim(x0[:, 0].min() - 1, x0[:, 0].max() + 1)
    plt.ylim(0, 2)
    plt.title("ROA estimation in xz plane")
    plt.xlabel("x")
    plt.ylabel("z")
    plt.tight_layout()
    return plt


def getRho(plant, knots, uncertainty):

    knots = np.array([knots]).flatten()
    n = len(knots)
    rho = np.zeros(len(knots))

    for i, knot in enumerate(knots):
        rho[i] = plant.ROA(knot, uncertainty)
        print 'i: ', i, ', knot: ', knot, ', rho: ', rho[i]

    return rho


def urdfViz(plant):
    tree = RigidBodyTree(FindResource("/notebooks/6832-code/ben_uav.urdf"),
                         FloatingBaseType.kRollPitchYaw)

    vis = PlanarRigidBodyVisualizer(tree, xlim=[-1, 15], ylim=[-0.5, 2.5])
    tf = plant.ttraj[-1]
    times = np.linspace(0, tf, 200)
    posn = np.zeros((13, len(times)))
    for i, t in enumerate(times):
        x = plant.xdtraj_poly.value(t)
        u = plant.udtraj_poly.value(t)
        posn[0:6, i] = 0
        posn[6, i] = (x[0] - plant.xdtraj[:,0].min()) / 14  # x
        posn[7, i] = 0  # z
        posn[8, i] = x[1]  # y
        posn[9, i] = 0  # yz plane
        posn[10, i] = np.pi / 2 - x[4]  # pitch down
        posn[11, i] = 0  # yaw
        posn[12, i] = -u[1]  # tail angle

    test_poly = PiecewisePolynomial.FirstOrderHold(times, posn)
    return vis.animate(test_poly, repeat=False)

def plotMultipleTraj(plants):

    fig = plt.figure(figsize=(12,2.5))

    for plant in plants:
        plt.scatter(plant.xdtraj[:, 0] - plant.xdtraj[:,0].max(), plant.xdtraj[:, 1], c=plant.xdtraj[:,2]/plant.xdtraj[:,2].max(),
                    edgecolors='none', s=8, cmap='YlGn')

    plt.xlim((plants[0].xdtraj[:,0].min()-1-plants[0].xdtraj[:,0].max(), 1))
    plt.ylim((0, 2))
    plt.title('trajectories in x-z plane')
    plt.xlabel('x [m]')
    plt.ylabel('z [m]')
    cb = plt.colorbar()
    cb.set_label('$V/V_{max}$')
    plt.tight_layout()
    plt.show()

def animateMultipleTraj(plants, saveprefix=None):

    fig = plt.figure(figsize=(12, 2.5))
    xmax = plants[0].xdtraj[:, 0].max()
    ax = plt.axes(xlim=(plants[0].xdtraj[:, 0].min() - 1 - xmax, 1), ylim=(0, 2))

    tmax = 0.0
    i_tmax = 0
    tdata = []
    xdata = []
    vdata = []

    plt.title('trajectories in x-z plane')
    plt.xlabel('x [m]')
    plt.ylabel('z [m]')

    for i, plant in enumerate(plants):
        tdata.append(plant.ttraj)
        xdata.append(plant.xdtraj[:,0] - xmax)
        vdata.append(plant.xdtraj[:,2] / plant.xdtraj[:,2].max())
        if plant.ttraj[-1] > tmax:
            tmax = plant.ttraj[-1]
            i_tmax = i

    for i, ti in enumerate(tdata):
        tdata[i] = ti - ti.max() + tmax

    for frame in range(1, len(plants[0].ttraj)):

        tlim = tdata[i_tmax][frame]
        for i, plant in enumerate(plants):
            if tdata[i][-1] >= tlim:
                i_tlim = np.argmax(tdata[i] >= tlim)
                sp = ax.scatter(xdata[i][:i_tlim], plant.xdtraj[:i_tlim,1], c=vdata[i][:i_tlim], edgecolors='none', s=8, cmap='YlGn')

        # if frame == 1:
        #     cb = plt.colorbar(sp)
        #     cb.set_label('$V/V_{max}$')
        if saveprefix is not None:
            filename = (saveprefix + str(frame).zfill(3) + '.png')
            plt.savefig(filename, dpi=300, bbox_inches="tight")

        fig.canvas.draw()

