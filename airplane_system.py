import numpy as np

from pydrake.math import sin, cos
from pydrake.trajectories import PiecewisePolynomial
from pydrake.solvers._mathematicalprogram_py import MathematicalProgram, SolutionResult, SolverType
from pydrake.systems.framework import VectorSystem
from pydrake.systems.controllers import LinearQuadraticRegulator
from pydrake._symbolic_py import Variables

import time
from scipy.misc import derivative


# Formulate fixed-wing UAV as Drake VectorSystem
class AirplaneSystem(VectorSystem):
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

        self.mp_result = None
        self.K = None
        self.S_lqr = None

    def _DoCalcVectorTimeDerivatives(self, context, u, x, xdot):
        # just a wrapper in case I use Drake for more things
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
        # 6D state (x, z, V, gamma, theta, q)
        # 2D input (thrust [N], elevator [deg])

        x = state[0]
        z = state[1]
        V = state[2]
        gamma = state[3]
        theta = state[4]
        q = state[5]

        FT = u[0]
        de = u[1]

        cg = cos(gamma)
        sg = sin(gamma)
        alpha = theta - gamma
        ca = cos(alpha)
        sa = sin(alpha)

        m = self.m
        g = self.g
        S = self.S
        rho = self.rho  # assume stays near sea level
        cbar = self.cbar
        I_y = self.I_y
        if uncertainty:
            m = 1.05 * m
            S = 0.95 * S
            I_y = 1.05 * I_y

        # assume constant aero moment coeffs
        CM_alpha = self.CM_alpha
        CM_q = self.CM_q
        CM_de = self.CM_de

        # CL, CD based on flat plate theory
        CL = 2 * ca * sa
        CD = 2 * sa ** 2
        CM = CM_alpha * alpha + CM_q * q + CM_de * de

        L = (0.5 * rho * S * V ** 2) * CL
        D = (0.5 * rho * S * V ** 2) * CD
        M = (0.5 * rho * S * V ** 2) * cbar * CM

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
        # do trajectory optimization. either direct transcription or direct collocation

        # stopwatch for solver time
        tsolve_pre = time.time()

        (x_goal, V_goal, gamma_goal, q_goal) = (200.0, state_initial[2], 0.0, 0.0)

        # time intervals for optimization discretization
        if not dircol:
            # N = 140
            N = int(np.floor(0.8 * np.abs(x_goal - state_initial[0])))
        else:
            N = 30

        # optimization problem: variables t_f, u[k], x[k]
        mp = MathematicalProgram()

        t_f = mp.NewContinuousVariables(1, "t_f")
        # time_array = t_f[0] * np.linspace(0.0, 1.0, (N + 1))

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
            #             return dt * (50 * u0.dot(u0) + 10 * u1.dot(u1)) + X[0]
            return dt * (1.0 * u0.dot(u0) + 1.0 * u1.dot(u1)) + 1.0 * X[0] * (u0.dot(v))
            # return dt * (1.0 * u0.dot(u0) + 1.0 * u1.dot(u1) + 10.0 * X[0] * (u0.dot(v)))

        #             return 0.5*X[0]*(u0.dot(v))

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
        dt = t_f[0] / N
        if not dircol:
            # direct transcription
            for j in range(1, N + 1):
                #                 dt = time_array[j] - time_array[j-1]
                dynamic_prop = dt * self.airplaneLongDynamics(state_trajectory[j - 1, :], input_trajectory[j - 1, :])
                for k in range(len(state_initial)):
                    mp.AddConstraint(state_trajectory[j, k] == state_trajectory[j - 1, k] + dynamic_prop[k])
        else:
            # direct collocation
            for j in range(1, N + 1):
                #                 dt = time_array[j] - time_array[j-1]
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
            t_guess = (x_goal - state_initial[0]) / (0.5 * (V_goal + state_initial[2]))
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

        it_limit = int(max(20000, 40*mp.num_vars()))
        mp.SetSolverOption(SolverType.kSnopt, 'Iterations limit', it_limit)

        print("** solver begin with N = %d **" % N)
        # solve nonlinear optimization problem (w/SNOPT)
        result = mp.Solve()
        print result
        #         assert(result == SolutionResult.kSolutionFound)

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

        # save traj
        self.udtraj = input_trajectory
        self.xdtraj = state_trajectory_approx
        self.ttraj = time_array
        self.mp_result = result

        if not dircol:
            self.udtraj_poly = PiecewisePolynomial.FirstOrderHold(time_array[0:-1], input_trajectory.T)
        else:
            self.udtraj_poly = PiecewisePolynomial.FirstOrderHold(time_array, input_trajectory.T)
        #         self.xdtraj_poly = PiecewisePolynomial.FirstOrderHold(time_array, state_trajectory_approx.T)
        self.xdtraj_poly = PiecewisePolynomial.Cubic(time_array, state_trajectory_approx.T)

        return input_trajectory, state_trajectory_approx, time_array

    def trajOptRRT(self, state_initial, state_final, goal=False):
        # TODO: reconcile trajOpt and trajOptRRT (shouldn't take long)

        # stopwatch for solver time
        tsolve_pre = time.time()

        #         (x_goal, V_goal, gamma_goal, q_goal) = (200.0, state_initial[2], 0.0, 0.0)

        # time intervals for optimization discretization
        N = int(max([np.floor(0.8 * np.abs(state_final[0] - state_initial[0])), 6]))

        # optimization problem: variables t_f, u[k], x[k]
        mp = MathematicalProgram()

        t_f = mp.NewContinuousVariables(1, "t_f")

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
            #             return dt * (50 * u0.dot(u0) + 10 * u1.dot(u1)) + X[0]
            return dt * (1.0 * u0.dot(u0) + 1.0 * u1.dot(u1)) + 1.0 * X[0] * (u0.dot(v))

        #             return 0.5*X[0]*(u0.dot(v))

        mp.AddCost(totalcost, allvars)

        # initial state constraint
        for i in range(len(state_initial)):
            mp.AddLinearConstraint(state_trajectory[0, i] == state_initial[i])

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

        # dynamic constraints
        dt = t_f[0] / N
        # direct transcription
        for j in range(1, N + 1):
            #                 dt = time_array[j] - time_array[j-1]
            dynamic_prop = dt * self.airplaneLongDynamics(state_trajectory[j - 1, :], input_trajectory[j - 1, :])
            for k in range(len(state_initial)):
                mp.AddConstraint(state_trajectory[j, k] == state_trajectory[j - 1, k] + dynamic_prop[k])

        # initial guesses
        t_guess = (state_final[0] - state_initial[0]) / (0.5 * (state_final[2] + state_initial[2]))
        mp.SetInitialGuess(t_f[0], t_guess)

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

        for i in range(N):
            mp.SetInitialGuess(input_trajectory[i, 0], self.m * self.g / 3.5)
            mp.SetInitialGuess(input_trajectory[i, 1], 0.01)

        # time constraints
        mp.AddLinearConstraint(t_f[0] <= 2.0 * t_guess)
        mp.AddLinearConstraint(t_f[0] >= 0.5 * t_guess)

        it_limit = int(max(20000, 40 * mp.num_vars()))
        mp.SetSolverOption(SolverType.kSnopt, 'Iterations limit', it_limit)

        print("** solver begin with N = %d **" % N)
        # solve nonlinear optimization problem (w/SNOPT)
        result = mp.Solve()
        print result
        #         assert(result == SolutionResult.kSolutionFound)

        # convert from symbolic to float
        utraj = mp.GetSolution(input_trajectory)
        t_f = mp.GetSolution(t_f)
        xtraj = mp.GetSolution(state_trajectory)
        ttraj = t_f[0] * np.linspace(0.0, 1.0, (N + 1))

        tsolve_post = time.time()
        tsolve = tsolve_post - tsolve_pre

        solver_id = mp.GetSolverId()

        print ("** %s solver finished in %.1f seconds **\n" % (solver_id.name(), tsolve))
        print ("t_f computed: %.3f seconds" % t_f[0])

        # get total cost of solution
        if result == SolutionResult.kSolutionFound:
            thrust = utraj[:, 0]
            elev = utraj[:, 1]
            vel = xtraj[:, 2]
            allvars = np.hstack((t_f[0], thrust, elev, vel))
            print ("cost computed: %.3f" % totalcost(allvars))

        return utraj, xtraj, ttraj, result

    def calcTvlqrStabilization(self):
        if self.mp_result != SolutionResult.kSolutionFound:
            print "Optimal traj not computed"
            return None

        xd = self.xdtraj
        ud = self.udtraj

        ## V1
        # define Q, R for LQR cost
        # calc df/dx, df/du with numerical differentiation
        # calculate A(t), B(t) at knot points
        # calculate K(t) using A, B, Q, R

        n = self.num_states
        m = self.num_inputs
        k = ud.shape[0]  # time steps

        Q = np.eye(n)
        R = 0.1 * np.eye(m)
        K = np.zeros((k, m, n))
        S = np.zeros((k, n, n))

        for i in range(k):
            A = self.A(xd[i, :], ud[i, :])
            B = self.B(xd[i, :], ud[i, :])
            K[i], S[i] = LinearQuadraticRegulator(A, B, Q, R)

        self.K = K
        self.S_lqr = S
        return K

        ## V2
        # define Q, R for LQR cost
        # make polynomial splines of traj (u(t), x(t))
        # differentiate x(t) polynomial to get xdot(t)
        # calculate A(t), B(t)
        # calculate S(t) by solving ODE backwards in time
        # calculate K(t)

    def A(self, state, u):
        n = self.num_states
        A = np.zeros((n, n))
        for row in range(n):
            for col in range(n):
                def f(x):
                    f = self.airplaneLongDynamics(np.hstack((state[:col], x, state[col + 1:])), u, uncertainty=False)
                    return f[row]

                diff = derivative(f, state[col], dx=0.01)  # scipy finite difference
                A[row, col] = diff
        return A

    def B(self, state, u):
        n = self.num_states
        m = self.num_inputs
        B = np.zeros((n, m))
        for col in range(m):
            for row in range(n):
                def f(x):
                    f = self.airplaneLongDynamics(state, np.hstack((u[:col], x, u[col + 1:])), uncertainty=False)
                    return f[row]

                diff = derivative(f, u[col], dx=0.01)  # scipy finite difference
                B[row, col] = diff
        return B

    def Klookup(self, t):
        # Compute LQR feedback gain K at time t via linear interoplation between K at knot points
        idx_closest = (np.abs(self.ttraj - t)).argmin()

        # limit to valid indices
        if idx_closest >= (self.K.shape[0] - 1):
            return self.K[-1]
        elif idx_closest <= 0:
            return self.K[0]
        else:
            # linearly interoplate between two points
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
        # Compute LQR cost to go matrix S at time t via linear interpolation
        idx_closest = (np.abs(self.ttraj - t)).argmin()

        # limit to valid indices
        if idx_closest >= (self.S_lqr.shape[0] - 1):
            return self.S_lqr[-1]
        elif idx_closest <= 0:
            return self.S_lqr[0]
        else:
            # linearly interoplate between two points
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

    # second-order taylor expansion around x0 (elementwise)
    def secondOrderTaylorCL(self, x, x0, u0, K, unc):
        n = self.num_states
        xbar = x - x0
        # evaluate closed loop dynamics on nominal traj: (u = u0 - K xbar) with xbar = 0
        fcl_0 = self.airplaneLongDynamics(x0, u0, uncertainty=unc)
        dx = 0.1

        first_order = np.zeros_like(x)
        for i in range(n):
            # dfcl/dx(i1):
            for i1 in range(n):
                x_pos = np.array(x0)
                x_neg = np.array(x0)
                x_pos[i1] += dx
                x_neg[i1] -= dx

                f_pos = self.airplaneLongDynamics(x_pos, u0 - K.dot(x_pos - x0), uncertainty=unc)
                f_neg = self.airplaneLongDynamics(x_neg, u0 - K.dot(x_neg - x0), uncertainty=unc)
                dfcl_dx = (f_pos[i] - f_neg[i]) / (2.0 * dx)
                first_order[i] += dfcl_dx * xbar[i1]

        second_order = np.zeros_like(x)
        for i in range(n):
            for i1 in range(n):
                # d2fcl/dx(i1)dx(i2):
                for i2 in range(n):
                    x_p1p2 = np.array(x0)
                    x_p1n2 = np.array(x0)
                    x_n1p2 = np.array(x0)
                    x_n1n2 = np.array(x0)
                    x_p1p2[i1] += dx
                    x_p1p2[i2] += dx
                    x_p1n2[i1] += dx
                    x_p1n2[i2] -= dx
                    x_n1p2[i1] -= dx
                    x_n1p2[i2] += dx
                    x_n1n2[i1] -= dx
                    x_n1n2[i2] -= dx
                    f_p1p2 = self.airplaneLongDynamics(x_p1p2, u0 - K.dot(x_p1p2 - x0), uncertainty=unc)
                    f_p1n2 = self.airplaneLongDynamics(x_p1n2, u0 - K.dot(x_p1n2 - x0), uncertainty=unc)
                    f_n1p2 = self.airplaneLongDynamics(x_n1p2, u0 - K.dot(x_n1p2 - x0), uncertainty=unc)
                    f_n1n2 = self.airplaneLongDynamics(x_n1n2, u0 - K.dot(x_n1n2 - x0), uncertainty=unc)

                    d2fcl_dx1dx2 = (f_p1p2[i] + f_n1n2[i] - f_p1n2[i] - f_n1p2[i]) / (4.0 * dx ** 2)
                    second_order[i] += d2fcl_dx1dx2 * xbar[i1] * xbar[i2]

        return fcl_0 + first_order + 0.5 * second_order

    def SolveSOS_h(self, rho, knot, unc):

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
               2.0 * xbar.dot(S.dot(self.secondOrderTaylorCL(xbar + x0, x0, u0, K, unc)))

        # Vdot = 2.0 * xbar.dot(S.dot(self.Dynamics(x, u0-K.dot(xbar))))[0]

        (h, constraint) = mp.NewSosPolynomial(Variables(xbar), 2)

        mp.AddSosConstraint((V - rho) * h.ToExpression() - Vdot)
        sol_result = mp.Solve()

        if sol_result == SolutionResult.kSolutionFound:
            h_subs = mp.SubstituteSolution(h)
        else:
            # print 'no feasible h(xbar) found'
            h_subs = None

        return sol_result, h_subs

    #     def SolveSOS_rho(self, h, knot, unc):

    #         mp = MathematicalProgram()
    #         xbar = mp.NewIndeterminates(self.num_states, 'xbar')

    #         K = self.K[knot]
    #         S = self.S_lqr[knot]
    #         x0 = self.xdtraj[knot]
    #         u0 = self.udtraj[knot]

    #         dt = self.ttraj[2] - self.ttraj[1] # assume fixed

    #         if knot >= 1:
    #             Sdot = (self.S_lqr[knot] - self.S_lqr[knot-1]) / dt
    #         else:
    #             Sdot = (self.S_lqr[1] - self.S_lqr[0]) / dt

    #         V = xbar.dot(S.dot(xbar))
    #         Vdot = xbar.dot(Sdot.dot(xbar)) + \
    #                     2.0 * xbar.dot(S.dot(self.secondOrderTaylorCL(xbar + x0, x0, u0, K, unc)))

    #         rho = mp.NewContinuousVariables(1, 'rho')[0]

    #         mp.AddSosConstraint((V - rho) * h.ToExpression() - Vdot)
    #         mp.AddLinearCost(-rho)
    #         mp.AddLinearConstraint(rho >= 0)
    #         sol_result = mp.Solve()

    #         if sol_result == SolutionResult.kSolutionFound:
    #             rho = mp.GetSolution(rho)
    #         else:
    #             print 'no feasible rho found'
    #             rho = 0.0

    #         return sol_result, rho

    def ROA(self, knot, unc):
        rho = 0.01
        rho_max = 0.0

        if self.K is None:
            print 'computing TVLQR along trajectory'
            K = self.calcTvlqrStabilization()

        # alternate between finding feasible h(xbar) (which is SOS)
        # and maximizing rho for a given h(xbar)
        for i in range(15):
            if rho < 0.0001:
                return rho_max

            res_h, h = self.SolveSOS_h(rho, knot, unc)
            if res_h == SolutionResult.kSolutionFound:
                # program was feasible, maximize rho
                rho_max = max(rho, rho_max)
                #                 res_rho, rho = self.SolveSOS_rho(h, knot, unc)
                rho = 1.5 * rho
            else:
                # program wasn't feasible, make rho smaller
                rho = 0.75 * rho
            # print 'rho: ', rho
        return rho_max
