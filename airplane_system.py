import numpy as np

from pydrake.math import sin, cos
from pydrake.trajectories import PiecewisePolynomial
from pydrake.solvers._mathematicalprogram_py import MathematicalProgram, SolutionResult, SolverType
from pydrake.systems.framework import VectorSystem
from pydrake.systems.controllers import LinearQuadraticRegulator
from pydrake._symbolic_py import Variables, MonomialBasis, Variable, Jacobian

import time

class AirplaneSystem(VectorSystem):
    """
    Model the aircraft dynamics in a pydrake VectorSystem. This class also holds the methods for trajectory
    optimization, TVLQR stabilization, as well as region of attraction estimation via bilinear alternations
    of SOS programs.
    """
    def __init__(self):
        VectorSystem.__init__(self,
                              2,  # two inputs
                              6)  # six outputs
        self._DeclareContinuousState(6)  # six state variables

        self.num_states = 6  # number of states
        self.num_inputs = 2  # number of inputs

        self.g = 9.81  # gravity
        self.m = 0.15  # mass (kg)
        self.S = 0.12  # wing area (m^2)
        self.rho = 1.2  # air density ~SL
        self.cbar = 0.12  # mean chord
        self.I_y = 0.004  # pitching mom. of inertia

        # pitching moment coefficients
        self.CM_alpha = -0.08
        self.CM_q = -0.2
        self.CM_de = -0.04

        self.mp_result = None  # SolutionResult from trajOpt SNOPT program

        self.K = None      # LQR gain matrix at each knot point
        self.S_lqr = None  # LQR cost-to-go matrix at each knot point

        self.poly_order = 4  # order of polynomial for SOS ROA estimation

    def _DoCalcVectorTimeDerivatives(self, context, u, x, xdot):
        """
        Just wrapping airplane dynamics in case I use pydrake simulator.
        """
        xdot = self.airplaneLongDynamics(x, u)

    def _DoCalcVectorOutput(self, context, u, x, y):
        # all states measured
        y[:] = x

    def _DoHasDirectFeedthrough(self, input_port, output_port):
        # no direct feedthrough
        if input_port == 0 and output_port == 0:
            return False
        else:
            return None

    def airplaneLongDynamics(self, state, u, uncertainty=False):
        """
        Compute the longitudinal dynamics of the airplane, as in xdot = f(x,u). Can compute with/without uncertainty,
        where uncertainty=True is used for simulation of system with modeling errors. Accepts either pydrake::symbolic
        or floats for state, u, and formats output accordingly.

        state: 6D state (x, z, V, gamma, theta, q)
        u: 2D input (thrust [N], elevator [deg])
        """

        x = state[0]  # horizontal position
        z = state[1]  # vertical position
        V = state[2]  # airspeed
        gamma = state[3]  # flight path angle
        theta = state[4]  # pitch angle
        q = state[5]  # pitch rate

        FT = u[0]  # thrust force
        de = u[1]  # elevator deflection

        cg = cos(gamma)
        sg = sin(gamma)
        alpha = theta - gamma  # angle of attack
        ca = cos(alpha)
        sa = sin(alpha)

        m = self.m
        g = self.g
        S = self.S
        rho = self.rho  # assume stays near sea level
        cbar = self.cbar
        I_y = self.I_y
        if uncertainty:
            m = 1.05 * m  # make heavier
            S = 0.95 * S  # make wing area smaller
            I_y = 1.05 * I_y  # increase moment of inertia

        # assume constant aero moment coefficients
        CM_alpha = self.CM_alpha
        CM_q = self.CM_q
        CM_de = self.CM_de

        # CL, CD based on flat plate theory
        CL = 2 * ca * sa
        CD = 2 * sa ** 2
        CM = CM_alpha * alpha + CM_q * q + CM_de * de

        # lift force, drag force, moment
        L = (0.5 * rho * S * V ** 2) * CL
        D = (0.5 * rho * S * V ** 2) * CD
        M = (0.5 * rho * S * V ** 2) * cbar * CM

        if u.dtype == np.dtype('O'):
            derivs = np.zeros_like(state, dtype=object)
        else:
            derivs = np.zeros_like(state)

        # (dynamics from Stevens, Lewis, Johnson: "Aircraft Control and Simulation" CH2.5)
        # positions
        derivs[0] = state[2] * cg
        derivs[1] = state[2] * sg
        # vdot
        derivs[2] = (1.0 / m) * (-D + FT * ca - m * g * sg)
        # gammadot
        derivs[3] = (1.0 / (m * V)) * (L + FT * sa - m * g * cg)
        # thetadot
        derivs[4] = q
        # qdot
        derivs[5] = M / I_y

        return derivs

    def trajOpt(self, state_initial, dircol=0, second_pass=False):
        """
        Perform trajectory optimization, using either direct transcription or direct collocation.
        trajOptRRT() is neater and more useful -- just keeping this around to avoid rewriting sims for class project.
        """

        # stopwatch for solver time
        tsolve_pre = time.time()

        (x_goal, V_goal, gamma_goal, q_goal) = (200.0, state_initial[2], 0.0, 0.0)

        # number of knot points - proportional to x-distance seems to work well
        if not dircol:
            N = int(np.floor(0.8 * np.abs(x_goal - state_initial[0])))
        else:
            N = 30

        # optimization problem: variables t_f, u[k], x[k]
        mp = MathematicalProgram()

        t_f = mp.NewContinuousVariables(1, "t_f")
        dt = t_f[0] / N

        k = 0
        u = mp.NewContinuousVariables(2, "u_%d" % k)
        input_trajectory = u

        x = mp.NewContinuousVariables(6, "x_%d" % k)
        state_trajectory = x

        for k in range(1, N):
            u = mp.NewContinuousVariables(2, "u_%d" % k)
            x = mp.NewContinuousVariables(6, "x_%d" % k)
            input_trajectory = np.vstack((input_trajectory, u))
            state_trajectory = np.vstack((state_trajectory, x))

        x = mp.NewContinuousVariables(6, "x_%d" % N)
        state_trajectory = np.vstack((state_trajectory, x))

        # for dircol we can use u_N and first-order hold
        if dircol:
            u = mp.NewContinuousVariables(2, "u_%d" % N)
            input_trajectory = np.vstack((input_trajectory, u))

        print "Number of decision vars", mp.num_vars()

        # cost function: penalize time and control effort
        thrust = input_trajectory[:, 0]
        elev = input_trajectory[:, 1]
        vel = state_trajectory[:, 2]
        allvars = np.hstack((t_f[0], thrust, elev, vel))
        # TODO: use u of length n+1 for dircol
        def totalcost(X):
            dt = X[0] / N
            u0 = X[1:N + 1]
            u1 = X[N + 1:2 * N + 1]
            v = X[2 * N + 1:3 * N + 1]  # cut last item if dirtrans
            return dt * (1.0 * u0.dot(u0) + 1.0 * u1.dot(u1)) + 1.0 * X[0] * (u0.dot(v))
            # return dt * (1.0 * u0.dot(u0) + 1.0 * u1.dot(u1) + 10.0 * X[0] * (u0.dot(v)))

        mp.AddCost(totalcost, allvars)

        # initial state constraint
        for i in range(len(state_initial)):
            mp.AddLinearConstraint(state_trajectory[0, i] == state_initial[i])

        # final state constraint (x position)
        mp.AddLinearConstraint(state_trajectory[-1, 0] == x_goal)

        # final state constraint (z position) NOTE: range is acceptable
        mp.AddLinearConstraint(state_trajectory[-1, 1] <= 1.5)
        mp.AddLinearConstraint(state_trajectory[-1, 1] >= 0.5)

        # final state constraint (velocity) NOTE: range is acceptable
        mp.AddLinearConstraint(state_trajectory[-1, 2] <= 1.5 * V_goal)
        mp.AddLinearConstraint(state_trajectory[-1, 2] >= V_goal)

        # final state constraint (flight path angle) NOTE: small range here
        mp.AddLinearConstraint(state_trajectory[-1, 3] <= gamma_goal + 1.0 * np.pi / 180.0)
        mp.AddLinearConstraint(state_trajectory[-1, 3] >= gamma_goal - 1.0 * np.pi / 180.0)

        # final state constraint (pitch rate)
        mp.AddLinearConstraint(state_trajectory[-1, 5] == q_goal)

        # input constraints
        for i in range(len(input_trajectory[:, 0])):
            mp.AddLinearConstraint(input_trajectory[i, 0] >= 0.0)
            mp.AddLinearConstraint(input_trajectory[i, 0] <= 1.2 * self.m * self.g)
            mp.AddLinearConstraint(input_trajectory[i, 1] >= -30.0)
            mp.AddLinearConstraint(input_trajectory[i, 1] <= 30.0)

        # state constraints
        for i in range(len(state_trajectory[:, 0])):
            # x position
            mp.AddLinearConstraint(state_trajectory[i, 0] >= state_initial[0])
            mp.AddLinearConstraint(state_trajectory[i, 0] <= x_goal)
            # z position
            mp.AddLinearConstraint(state_trajectory[i, 1] >= 0.3)
            mp.AddLinearConstraint(state_trajectory[i, 1] <= 2.0)
            # velocity
            mp.AddLinearConstraint(state_trajectory[i, 2] >= 1.0)
            mp.AddLinearConstraint(state_trajectory[i, 2] <= 3.0 * state_initial[2])
            # flight path angle
            mp.AddLinearConstraint(state_trajectory[i, 3] >= -30.0 * np.pi / 180.0)
            mp.AddLinearConstraint(state_trajectory[i, 3] <= 30.0 * np.pi / 180.0)
            # pitch angle
            mp.AddLinearConstraint(state_trajectory[i, 4] >= -20.0 * np.pi / 180.0)
            mp.AddLinearConstraint(state_trajectory[i, 4] <= 40.0 * np.pi / 180.0)
            # pitch rate
            mp.AddLinearConstraint(state_trajectory[i, 5] >= -20.0 * np.pi / 180.0)
            mp.AddLinearConstraint(state_trajectory[i, 5] <= 20.0 * np.pi / 180.0)

        # dynamic constraints
        if not dircol:
            # direct transcription
            for j in range(1, N + 1):
                dynamic_prop = dt * self.airplaneLongDynamics(state_trajectory[j - 1, :], input_trajectory[j - 1, :])
                for k in range(len(state_initial)):
                    mp.AddConstraint(state_trajectory[j, k] == state_trajectory[j - 1, k] + dynamic_prop[k])
        else:
            # direct collocation
            for j in range(1, N + 1):
                x0 = state_trajectory[j - 1, :]
                x1 = state_trajectory[j, :]
                xdot0 = self.airplaneLongDynamics(x0, input_trajectory[j - 1, :])
                xdot1 = self.airplaneLongDynamics(x1, input_trajectory[j, :])

                xc = 0.5 * (x1 + x0) + dt * (xdot0 - xdot1) / 8.0
                xdotc = - 1.5 * (x0 - x1) / dt - 0.25 * (xdot0 + xdot1)
                uc = 0.5 * (input_trajectory[j - 1, :] + input_trajectory[j, :])
                f_xc = self.airplaneLongDynamics(xc, uc)
                for k in range(len(state_initial)):
                    # TODO: why does "==" cause "kUnknownError"?
                    #                     mp.AddConstraint(xdotc[k] - f_xc[k] == 0.0)
                    mp.AddConstraint(xdotc[k] <= f_xc[k] + 0.001)
                    mp.AddConstraint(xdotc[k] >= f_xc[k] - 0.001)

        # allow for warm start of dircol program with output of dirtrans program
        if (second_pass) and (self.mp_result == SolutionResult.kSolutionFound):
            # warm start using previous output
            print 'warm start to traj opt'
            t_guess = self.ttraj[-1]
            mp.SetInitialGuess(t_f[0], t_guess)

            for i in range(len(state_trajectory[:, 0])):
                for j in range(len(state_initial)):
                    mp.SetInitialGuess(state_trajectory[i, j], self.xdtraj[i, j])
            for i in range(N):
                mp.SetInitialGuess(input_trajectory[i, 0], self.udtraj[i, 0])
                mp.SetInitialGuess(input_trajectory[i, 1], self.udtraj[i, 1])

            # time constraints
            mp.AddLinearConstraint(t_f[0] <= 1.25 * t_guess)
            mp.AddLinearConstraint(t_f[0] >= 0.8 * t_guess)

        else:
            # initial guesses
            t_guess = np.abs(x_goal - state_initial[0]) / (0.5 * (V_goal + state_initial[2]))
            mp.SetInitialGuess(t_f[0], t_guess)

            z_final_dummy = state_initial[1]
            theta_final_dummy = state_initial[4]
            state_final_dummy = np.array([x_goal, z_final_dummy, V_goal, gamma_goal, theta_final_dummy, q_goal])
            for i in range(len(state_trajectory[:, 0])):
                state_guess = ((N - i) / N) * state_initial + (i / N) * state_final_dummy
                for j in range(len(state_guess)):
                    mp.SetInitialGuess(state_trajectory[i, j], state_guess[j])

            for i in range(N):
                mp.SetInitialGuess(input_trajectory[i, 0], self.m * self.g / 3.5)
                mp.SetInitialGuess(input_trajectory[i, 1], 0.01)

            # time constraints
            mp.AddLinearConstraint(t_f[0] <= 2.0 * t_guess)
            mp.AddLinearConstraint(t_f[0] >= 0.5 * t_guess)

        # set SNOPT iteration limit
        it_limit = int(max(20000, 40*mp.num_vars()))
        mp.SetSolverOption(SolverType.kSnopt, 'Iterations limit', it_limit)

        print("** solver begin with N = %d **" % N)
        # solve nonlinear optimization problem (w/SNOPT)
        result = mp.Solve()
        print result

        # convert from symbolic to float
        input_trajectory = mp.GetSolution(input_trajectory)
        t_f = mp.GetSolution(t_f)
        state_trajectory_approx = mp.GetSolution(state_trajectory)
        time_array = t_f[0] * np.linspace(0.0, 1.0, (N + 1))

        tsolve_post = time.time()
        tsolve = tsolve_post - tsolve_pre

        solver_id = mp.GetSolverId()

        print ("** %s solver finished in %.1f seconds **\n" % (solver_id.name(), tsolve))
        print ("t_f computed: %.3f seconds" % t_f[0])

        # get total cost of solution
        if result == SolutionResult.kSolutionFound:
            thrust = input_trajectory[:, 0]
            elev = input_trajectory[:, 1]
            vel = state_trajectory_approx[:, 2]
            allvars = np.hstack((t_f[0], thrust, elev, vel))
            print ("cost computed: %.3f" % totalcost(allvars))

        # save traj (this is a bit sloppy and redundant but scripts for visualization currently rely on this)
        self.udtraj = input_trajectory
        self.xdtraj = state_trajectory_approx
        self.ttraj = time_array
        self.mp_result = result

        # save polynomials of input, state trajectories
        if not dircol:
            self.udtraj_poly = PiecewisePolynomial.FirstOrderHold(time_array[0:-1], input_trajectory.T)
        else:
            self.udtraj_poly = PiecewisePolynomial.FirstOrderHold(time_array, input_trajectory.T)
        self.xdtraj_poly = PiecewisePolynomial.Cubic(time_array, state_trajectory_approx.T)

        return input_trajectory, state_trajectory_approx, time_array

    def trajOptRRT(self, state_initial, state_final, goal=False, verbose=False):
        """
        Perform trajectory optimization via direct transcription, for trajectory from state_initial to state_final.
        If goal=True, conditions are relaxed on convergence to state_final since overall goal is to reach goal region
        and not necessarily a single point.
        """
        # TODO: reconcile trajOpt and trajOptRRT (shouldn't take long)

        # stopwatch for solver time
        tsolve_pre = time.time()

        # number of knot points - proportional to x-distance seems to work well
        N = int(max([np.floor(0.8 * np.abs(state_final[0] - state_initial[0])), 6]))

        # optimization problem: variables t_f, u[k], x[k]
        mp = MathematicalProgram()

        # variable for time to reach goal
        t_f = mp.NewContinuousVariables(1, "t_f")
        dt = t_f[0] / N

        k = 0
        u = mp.NewContinuousVariables(2, "u_%d" % k)
        input_trajectory = u

        x = mp.NewContinuousVariables(6, "x_%d" % k)
        state_trajectory = x

        for k in range(1, N):
            u = mp.NewContinuousVariables(2, "u_%d" % k)
            x = mp.NewContinuousVariables(6, "x_%d" % k)
            input_trajectory = np.vstack((input_trajectory, u))
            state_trajectory = np.vstack((state_trajectory, x))

        x = mp.NewContinuousVariables(6, "x_%d" % N)
        state_trajectory = np.vstack((state_trajectory, x))

        if verbose:
            print "Number of decision vars", mp.num_vars()

        # cost function: penalize electric energy use and overall control effort
        thrust = input_trajectory[:, 0]
        elev = input_trajectory[:, 1]
        vel = state_trajectory[:, 2]
        allvars = np.hstack((t_f[0], thrust, elev, vel))
        def totalcost(X):
            dt = X[0] / N
            u0 = X[1:N + 1]
            u1 = X[N + 1:2 * N + 1]
            v = X[2 * N + 1:3 * N + 1]
            return dt * (1.0 * u0.dot(u0) + 1.0 * u1.dot(u1) + 10.0 * X[0] * (u0.dot(v)))

        mp.AddCost(totalcost, allvars)

        # initial state constraint
        for i in range(len(state_initial)):
            mp.AddLinearConstraint(state_trajectory[0, i] == state_initial[i])

        # final state constraints
        if goal:
            # final state constraint (x position)
            mp.AddLinearConstraint(state_trajectory[-1, 0] == state_final[0])

            # final state constraint (z position) NOTE: range is acceptable
            mp.AddLinearConstraint(state_trajectory[-1, 1] <= 1.5)
            mp.AddLinearConstraint(state_trajectory[-1, 1] >= 0.5)

            # final state constraint (velocity) NOTE: range is acceptable
            mp.AddLinearConstraint(state_trajectory[-1, 2] <= 1.5 * state_final[2])
            mp.AddLinearConstraint(state_trajectory[-1, 2] >= state_final[2])

            # final state constraint (flight path angle) NOTE: small range here
            mp.AddLinearConstraint(state_trajectory[-1, 3] <= state_final[3] + 1.0 * np.pi / 180.0)
            mp.AddLinearConstraint(state_trajectory[-1, 3] >= state_final[3] - 1.0 * np.pi / 180.0)

            # final state constraint (pitch rate)
            mp.AddLinearConstraint(state_trajectory[-1, 5] == state_final[5])
        else:
            for i in range(len(state_initial)):
                mp.AddLinearConstraint(state_trajectory[-1, i] == state_final[i])

        # input constraints
        for i in range(len(input_trajectory[:, 0])):
            mp.AddLinearConstraint(input_trajectory[i, 0] >= 0.0)
            mp.AddLinearConstraint(input_trajectory[i, 0] <= 1.2 * self.m * self.g)
            mp.AddLinearConstraint(input_trajectory[i, 1] >= -30.0)
            mp.AddLinearConstraint(input_trajectory[i, 1] <= 30.0)

        # state constraints
        for i in range(len(state_trajectory[:, 0])):
            # x position
            mp.AddLinearConstraint(state_trajectory[i, 0] >= state_initial[0])
            mp.AddLinearConstraint(state_trajectory[i, 0] <= state_final[0])
            # z position
            mp.AddLinearConstraint(state_trajectory[i, 1] >= 0.3)
            mp.AddLinearConstraint(state_trajectory[i, 1] <= 2.0)
            # velocity
            mp.AddLinearConstraint(state_trajectory[i, 2] >= 2.0)
            mp.AddLinearConstraint(state_trajectory[i, 2] <= 18.0)
            # flight path angle
            mp.AddLinearConstraint(state_trajectory[i, 3] >= -30.0 * np.pi / 180.0)
            mp.AddLinearConstraint(state_trajectory[i, 3] <= 30.0 * np.pi / 180.0)
            # pitch angle
            mp.AddLinearConstraint(state_trajectory[i, 4] >= -20.0 * np.pi / 180.0)
            mp.AddLinearConstraint(state_trajectory[i, 4] <= 40.0 * np.pi / 180.0)
            # pitch rate
            mp.AddLinearConstraint(state_trajectory[i, 5] >= -20.0 * np.pi / 180.0)
            mp.AddLinearConstraint(state_trajectory[i, 5] <= 20.0 * np.pi / 180.0)

        # dynamic constraints (direct transcription)
        for j in range(1, N + 1):
            dynamic_prop = dt * self.airplaneLongDynamics(state_trajectory[j - 1, :], input_trajectory[j - 1, :])
            for k in range(len(state_initial)):
                mp.AddConstraint(state_trajectory[j, k] == state_trajectory[j - 1, k] + dynamic_prop[k])

        # initial guess for time
        t_guess = np.abs(state_final[0] - state_initial[0]) / (0.5 * (state_final[2] + state_initial[2]))
        mp.SetInitialGuess(t_f[0], t_guess)

        # initial guesses for state
        if goal:
            state_final_dummy = np.array(state_final)
            state_final_dummy[1] = state_initial[1]
            state_final_dummy[4] = state_initial[4]
            for i in range(len(state_trajectory[:, 0])):
                state_guess = ((N - i) / N) * state_initial + (i / N) * state_final_dummy
                for j in range(len(state_guess)):
                    mp.SetInitialGuess(state_trajectory[i, j], state_guess[j])
        else:
            for i in range(len(state_trajectory[:, 0])):
                state_guess = ((N - i) / N) * state_initial + (i / N) * state_final
                for j in range(len(state_guess)):
                    mp.SetInitialGuess(state_trajectory[i, j], state_guess[j])

        # initial guesses for input
        for i in range(N):
            mp.SetInitialGuess(input_trajectory[i, 0], self.m * self.g / 3.5)
            mp.SetInitialGuess(input_trajectory[i, 1], 0.01)

        # time constraints
        mp.AddLinearConstraint(t_f[0] <= 2.0 * t_guess)
        mp.AddLinearConstraint(t_f[0] >= 0.5 * t_guess)

        # set SNOPT iteration limit
        it_limit = int(max(20000, 40 * mp.num_vars()))
        mp.SetSolverOption(SolverType.kSnopt, 'Iterations limit', it_limit)

        if verbose:
            print("** solver begin with N = %d **" % N)
        # solve nonlinear optimization problem (w/SNOPT)
        result = mp.Solve()
        if verbose:
            print result

        # convert from symbolic to float
        utraj = mp.GetSolution(input_trajectory)
        t_f = mp.GetSolution(t_f)
        xtraj = mp.GetSolution(state_trajectory)
        ttraj = t_f[0] * np.linspace(0.0, 1.0, (N + 1))

        tsolve_post = time.time()
        tsolve = tsolve_post - tsolve_pre

        solver_id = mp.GetSolverId()

        if verbose:
            print ("** %s solver finished in %.1f seconds **\n" % (solver_id.name(), tsolve))
            print ("t_f computed: %.3f seconds" % t_f[0])

        cost = -1
        # get total cost of solution
        if result == SolutionResult.kSolutionFound:
            thrust = utraj[:, 0]
            elev = utraj[:, 1]
            vel = xtraj[:, 2]
            allvars = np.hstack((t_f[0], thrust, elev, vel))
            cost = totalcost(allvars)
            if verbose:
                print ("cost computed: %.3f" % cost)

        return utraj, xtraj, ttraj, result, cost

    def calcTvlqrStabilization(self):
        """
        Compute time-varying LQR stabilization around trajectory. Linearize with nominal state, input at each
        knot point and compute LQR based on that linearization.
        """
        if self.mp_result != SolutionResult.kSolutionFound:
            print "Optimal traj not yet computed, can't stablize"
            return None

        xd = self.xdtraj
        ud = self.udtraj

        # # V1 (used here)
        # define Q, R for LQR cost
        # calc df/dx, df/du with pydrake autodiff
        # calculate A(t), B(t) at knot points
        # calculate K(t) using A, B, Q, R

        # # V2 (to implement if needed)
        # define Q, R for LQR cost
        # calc df/dx, df/du with pydrake autodiff
        # calculate A(t), B(t)
        # calculate S(t) by solving ODE backwards in time, potentially w/sqrt method
        # calculate K(t)

        n = self.num_states
        m = self.num_inputs
        k = ud.shape[0]  # time steps

        Q = np.eye(n)  # quadratic state cost
        R = 0.1 * np.eye(m)  # quadratic input cost
        K = np.zeros((k, m, n))
        S = np.zeros((k, n, n))

        for i in range(k):
            A = self.A(xd[i, :], ud[i, :])
            B = self.B(xd[i, :], ud[i, :])
            K[i], S[i] = LinearQuadraticRegulator(A, B, Q, R)

        self.K = K
        self.S_lqr = S
        return K


    def A(self, state, u):
        """
        Compute linearized state matrix (xdot = Ax + Bu), using pydrake::symbolic tools (Jacobian)
        """
        n = self.num_states
        A = np.zeros((n, n))

        # propagate symbolic variables through dynamics
        x_sym = np.zeros(n, dtype=object)
        for i in range(len(x_sym)):
            x_sym[i] = Variable('x_%d' % i)
        f_sym = self.airplaneLongDynamics(x_sym, u, uncertainty=False)

        # take Jacobian of f_sym w.r.t x_sym and evaluate at state
        dfcl_dx = Jacobian(f_sym, x_sym)
        env = dict(zip(x_sym, state))  # set pydrake::symbolic environment to evaluate
        for i in range(n):
            for j in range(n):
                A[i, j] = dfcl_dx[i, j].Evaluate(env)
        return A

    def B(self, state, u):
        """
        Compute linearized input matrix (xdot = Ax + Bu), using pydrake::symbolic tools (Jacobian)
        """
        n = self.num_states
        m = self.num_inputs
        B = np.zeros((n, m))

        # propagate symbolic variables through dynamics
        u_sym = np.zeros(m, dtype=object)
        for i in range(len(u_sym)):
            u_sym[i] = Variable('u_%d' % i)
        f_sym = self.airplaneLongDynamics(state, u_sym, uncertainty=False)

        # take Jacobian of f_sym w.r.t u_sym and evaluate at u
        dfcl_du = Jacobian(f_sym, u_sym)
        env = dict(zip(u_sym, u))  # set pydrake::symbolic environment to evaluate
        for i in range(n):
            for j in range(m):
                B[i, j] = dfcl_du[i, j].Evaluate(env)
        return B

    def Klookup(self, t):
        """
        Compute LQR feedback gain K at time t via linear interpolation between K at knot points. This could be
        replace by using Drake's PiecewisePolynomial but this works fine.
        """
        idx_closest = (np.abs(self.ttraj - t)).argmin()

        # limit to valid indices
        if idx_closest >= (self.K.shape[0] - 1):
            return self.K[-1]
        elif idx_closest <= 0:
            return self.K[0]
        else:
            # linearly interpolate between two points
            t_closest = self.ttraj[idx_closest]
            if t_closest > t:
                dt = self.ttraj[idx_closest] - self.ttraj[idx_closest - 1]
                return (t_closest - t) * self.K[idx_closest - 1] / dt + (t + dt - t_closest) * self.K[idx_closest] / dt
            elif t > t_closest:
                dt = self.ttraj[idx_closest + 1] - self.ttraj[idx_closest]
                return (t - t_closest) * self.K[idx_closest + 1] / dt + (t_closest + dt - t) * self.K[idx_closest] / dt
            else:
                return self.K[idx_closest]

    def Slookup(self, t):
        """
        Compute LQR cost-to-go matrix S at time t via linear interpolation between S at knot points. This could be
        replace by using Drake's PiecewisePolynomial but this works fine.
        """
        idx_closest = (np.abs(self.ttraj - t)).argmin()

        # limit to valid indices
        if idx_closest >= (self.S_lqr.shape[0] - 1):
            return self.S_lqr[-1]
        elif idx_closest <= 0:
            return self.S_lqr[0]
        else:
            # linearly interpolate between two points
            t_closest = self.ttraj[idx_closest]
            if t_closest > t:
                dt = self.ttraj[idx_closest] - self.ttraj[idx_closest - 1]
                return (t_closest - t) * self.S_lqr[idx_closest - 1] / dt + (t + dt - t_closest) * self.S_lqr[
                    idx_closest] / dt
            elif t > t_closest:
                dt = self.ttraj[idx_closest + 1] - self.ttraj[idx_closest]
                return (t - t_closest) * self.S_lqr[idx_closest + 1] / dt + (t_closest + dt - t) * self.S_lqr[
                    idx_closest] / dt
            else:
                return self.S_lqr[idx_closest]

    def PolynomialDynamics(self, xbar, x0, u0, K, unc):
        """
        Third-order Taylor expansion of dynamics around x0, u0. Works quite well with third-order. Uses pydrake
        symbolic tools to get first, second, third-order mixed partial derivatives and evaluate them at x0, u0.
        """
        n = self.num_states

        if xbar.dtype == np.dtype('O'):
            xbar_sym = xbar
        else:
            xbar_sym = np.zeros(n, dtype=object)  # to store Variables
            for i in range(len(xbar_sym)):
                xbar_sym[i] = Variable('x_%d' % i)

        x_sym = xbar_sym + x0
        u_sym = u0 - K.dot(xbar_sym)

        # evaluate closed loop dynamics on nominal traj: (u = u0 - K xbar) with xbar = 0
        poly_approx = np.zeros(n, dtype=object)
        poly_approx += self.airplaneLongDynamics(x0, u0, uncertainty=unc)  # initialize as f(x0, u0)

        f_sym = self.airplaneLongDynamics(x_sym, u_sym, uncertainty=unc)
        dfcl_dx_sym = Jacobian(f_sym, xbar_sym)
        dfcl_dx = np.zeros((n, n))
        env_0 = dict(zip(xbar_sym, np.zeros_like(x0)))
        for row in range(n):
            for col in range(n):
                dfcl_dx[row, col] = dfcl_dx_sym[row, col].Evaluate(env_0)

        for i in range(n):
            # first-order:
            poly_approx[i] += dfcl_dx[i, :].dot(xbar_sym)

            # second-order:
            hess_sym = Jacobian(dfcl_dx_sym[i, :], xbar_sym)
            hess = np.zeros((n, n))
            for row in range(n):
                for col in range(n):
                    hess[row, col] = hess_sym[row, col].Evaluate(env_0)
            poly_approx[i] += 0.5 * xbar_sym.dot(hess.dot(xbar_sym))

            # third-order:
            for x1 in range(n):
                d3_sym = Jacobian(hess_sym[x1, :], xbar_sym)
                d3 = np.zeros((n, n))
                for row in range(n):
                    for col in range(n):
                        d3[row, col] = d3_sym[row, col].Evaluate(env_0)
                poly_approx[i] += xbar_sym.dot(d3.dot(xbar_sym)) * xbar_sym[x1] / 6.0

        if xbar.dtype == np.dtype('O'):
            return poly_approx
        else:
            env = dict(zip(xbar_sym, xbar))
            to_return = np.zeros_like(x0)
            for i, el in enumerate(poly_approx):
                to_return[i] = el.Evaluate(env)
            return to_return

    # def SolveSOS_h(self, rho, knot, unc, rho_next, knot_next):
    def SolveSOS_h(self, rho, knot, unc):
        """
        One step of the bilinear alternations for TVLQR region of attraction estimation using sums-of-squares programs.
        This one takes a float for rho, and searches for a feasible SOS polynomial which maximizes an additive slack
        variable (gamma>0) on the main SOS constraint for the ROA estimation.
        If a feasible SOS polynomial was found it returns the numerical coefficient matrix Q of the polynomial (M' Q M).

        Note: still debugging issues with SCS solver which cause it to return 'SCS solved inaccurate' or
        'SCS infeasible inaccurate'.

        Note: this is currently not a fully correct implementation, as rho_dot is not being included in the SOS
        constraint while debugging issues.
        """
        mp = MathematicalProgram()
        xbar = mp.NewIndeterminates(self.num_states, 'xbar')

        K = self.K[knot]
        S = self.S_lqr[knot]
        x0 = self.xdtraj[knot]
        u0 = self.udtraj[knot]

        dt = self.ttraj[2] - self.ttraj[1]  # assume fixed

        if knot >= 1:
            Sdot = (self.S_lqr[knot] - self.S_lqr[knot - 1]) / dt
        else:
            Sdot = (self.S_lqr[1] - self.S_lqr[0]) / dt

        V = xbar.dot(S.dot(xbar))
        Vdot = xbar.dot(Sdot.dot(xbar)) + \
               2.0 * xbar.dot(S.dot(self.PolynomialDynamics(xbar, x0, u0, K, unc)))
        # Vdot = 2.0 * xbar.dot(S.dot(self.Dynamics(x, u0-K.dot(xbar))))[0]

        (h, constraint) = mp.NewSosPolynomial(Variables(xbar), self.poly_order)

        # if rho_next is not None:
        #     rhodot = (rho_next - rho) / (dt * np.abs(knot_next - knot))
        # else:
        #     rhodot = 0.0
        # mp.AddSosConstraint((V - rho) * h.ToExpression() - Vdot + rhodot)

        gamma = mp.NewContinuousVariables(1, 'gamma')[0]

        mp.AddSosConstraint((V - rho) * h.ToExpression() - Vdot - gamma)
        mp.AddLinearCost(-gamma)
        mp.AddLinearConstraint(gamma >= 0.0)
        sol_result = mp.Solve()

        if sol_result == SolutionResult.kSolutionFound:

            mb_ = MonomialBasis(h.indeterminates(), self.poly_order/2)
            Qdim = mb_.shape[0]

            coeffs = h.decision_variables()
            Q = np.zeros((Qdim, Qdim))
            row = 0
            col = 0
            for i, coeff in enumerate(coeffs):
                Q[row, col] = mp.GetSolution(coeff)
                Q[col, row] = mp.GetSolution(coeff)

                if col == Q.shape[0] - 1:
                    row += 1
                    col = row
                else:
                    col += 1
        else:
            # print 'no feasible h(xbar) found'
            Q = None

        return sol_result, Q

    def SolveSOS_rho(self, Q, knot, unc):
        """
        Second step of the bilinear alternations for TVLQR region of attraction estimation using sums-of-squares programs.
        This one takes a coefficient matrix Q and reconstructs the SOS polynomial M'QM by recreating the monomial basis
        M with the correct variables.
        It then tries to maximize the variable rho in the main SOS constraint for the given h=M'QM. If a feasible rho>0
        is found, it returns this.

        Note: still debugging issues with SCS solver which cause it to return 'SCS solved inaccurate' or
        'SCS infeasible inaccurate'.

        Note: this is currently not a fully correct implementation, as rho_dot is not being included in the SOS
        constraint while debugging issues.
        """
        mp = MathematicalProgram()
        xbar = mp.NewIndeterminates(self.num_states, 'xbar')

        K = self.K[knot]
        S = self.S_lqr[knot]
        x0 = self.xdtraj[knot]
        u0 = self.udtraj[knot]

        dt = self.ttraj[2] - self.ttraj[1] # assume fixed

        if knot >= 1:
            Sdot = (self.S_lqr[knot] - self.S_lqr[knot-1]) / dt
        else:
            Sdot = (self.S_lqr[1] - self.S_lqr[0]) / dt

        MB = MonomialBasis(Variables(xbar), self.poly_order/2)

        h = 0
        for i, mi in enumerate(MB):
            for j, mj in enumerate(MB):
                h += mi.ToExpression() * Q[i, j] * mj.ToExpression()

        # print 'h in SolveSOS_rho: ', h
        V = xbar.dot(S.dot(xbar))
        Vdot = xbar.dot(Sdot.dot(xbar)) + \
                    2.0 * xbar.dot(S.dot(self.PolynomialDynamics(xbar, x0, u0, K, unc)))

        rho = mp.NewContinuousVariables(1, 'rho')[0]

        mp.AddSosConstraint((V - rho) * h - Vdot)
        mp.AddLinearCost(-rho)
        mp.AddLinearConstraint(rho >= 0.0)
        sol_result = mp.Solve()

        if sol_result == SolutionResult.kSolutionFound:
            rho = mp.GetSolution(rho)
        else:
            print 'no feasible rho found'
            rho = 0.0

        return sol_result, rho

    # def ROA(self, knot, unc, rho_next, knot_next):
    def ROA(self, knot, unc):
        """
        Estimate the region of attraction 'funnel' around the optimal trajectory stabilized using TVLQR, by computing
        a volume (X | V(X, t) <= rho(t)) such that within this funnel, the vehicle will reach a bounded region around
        the goal state at t_f.

        This is done by alternating between two SOS programs. Due to issues still being debugged with the SOS
        maximization of rho, a rough estimation can be done by running the first SOS program and trying again with a
        larger/smaller value of rho based on the results, until rho converges.

        Note: this is currently not a fully correct implementation, as rho_dot is not being included in the SOS
        constraint while debugging issues.
        """
        rho = 0.01
        rho_max = 0.0

        if self.K is None:
            print 'computing TVLQR along trajectory'
            K = self.calcTvlqrStabilization()

        # alternate between finding feasible h(xbar) (which is SOS)
        # and maximizing rho for a given h(xbar)
        for i in range(15):
            # if rho < 0.0001:
            #     return rho_max

            # res_h, h = self.SolveSOS_h(rho, knot, unc, rho_next, knot_next)
            res_h, Q = self.SolveSOS_h(rho, knot, unc)
            if res_h == SolutionResult.kSolutionFound:
                # program was feasible, maximize rho
                res_rho, rho_opt = self.SolveSOS_rho(Q, knot, unc)
                if res_rho == SolutionResult.kSolutionFound and rho_opt >= 0:
                    rho_max = max(rho_opt, rho_max)
                    rho = rho_opt
                else:
                    rho = 0.75 * rho
                # rho = 1.5 * rho
            else:
                # program wasn't feasible, make rho smaller
                rho = 0.75 * rho
            print 'rho: ', rho
        return rho_max
