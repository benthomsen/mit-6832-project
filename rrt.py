from pydrake.solvers._mathematicalprogram_py import SolutionResult
from pydrake.trajectories import PiecewisePolynomial
import numpy as np
import time

class RRT(object):
    def __init__(self, x0, xG, plant, obs):
        self.x0 = x0
        self.xG = xG
        self.plant = plant
        self.obs = obs
        self.nodes = list()

    def run(self):
        # do RRT
        t_start = time.time()

        # maximum number of nodes
        N = 100

        # list holding all nodes in tree
        start_node = Node(self.x0)
        self.nodes.append(start_node)

        utraj, xtraj, ttraj, res = self.plant.trajOptRRT(self.x0, self.xG, goal=True)

        if res == SolutionResult.kSolutionFound:
            if self.collision_free(xtraj):
                print "goal reachable from start and obstacle free"
                self.nodes.append(Node(self.xG, parent_node=start_node))
                return self.nodes
            else:
                print "goal reachable from start but not obstacle free"
        else:
            print "goal not reachable from start"

        for i in range(N):
            print "i: ", i

            # get a random point in bounds that doesn't interesect with obstacles
            x_rand = self.sample_point()

            # find nearest existing node in tree
            x_near_node = self.nearest(x_rand)
            if x_near_node is not None:
                x_near = x_near_node.state
                # check for dynamically feasible path
                utraj, xtraj, ttraj, res = self.plant.trajOptRRT(x_near, x_rand, goal=False)

                if res == SolutionResult.kSolutionFound:
                    if self.collision_free(xtraj):
                        new_node = Node(x_rand, parent_node=x_near_node)
                        self.nodes.append(new_node)

                        # check whether we have arrived at the goal region
                        utraj, xtraj, ttraj, res = self.plant.trajOptRRT(x_rand, self.xG, goal=True)
                        if res == SolutionResult.kSolutionFound:
                            if self.collision_free(xtraj):
                                self.nodes.append(Node(self.xG, parent_node=new_node))
                                print('RRT path to goal found')
                                return self.nodes
                            else:
                                continue
                        else:
                            continue
                    else:
                        continue
                else:
                    continue
            else:
                continue

        t_finish = time.time()
        t_diff = t_finish - t_start
        print('RRT execution time: %.3f seconds' % t_diff)
        return self.nodes

    def collision_free(self, xtraj):
        # check for collisions along traj
        for xt in xtraj:
            for ob in self.obs:
                if xt[0] > ob[0] and xt[0] < (ob[0] + ob[2]) and xt[1] > ob[1] and xt[1] < (ob[1] + ob[3]):
                    return False
        return True

    def nearest(self, test_pt):
        # find nearest node in tree for connection (just using euclidian dist)
        # dists = np.zeros(len(self.nodes))
        # for i, node in enumerate(self.nodes):
        #     if node.state[0] < (test_pt[0] - 3.0):
        #         dists[i] = (node.state[0] - test_pt[0]) ** 2 + (node.state[1] - test_pt[1]) ** 2
        #
        # node_min = dists.argmin()

        # TODO: make this non-uniform
        ind_min = np.random.randint(0, len(self.nodes))
        return self.nodes[ind_min]

    def sample_point(self):
        xtest = np.random.rand(self.plant.num_states)  # all uniform 0-1
        xtest[0] = xtest[0] * (self.xG[0] - self.x0[0]) + self.x0[0]
        xtest[1] = xtest[1] * (2.0 - 0.3) + 0.3
        xtest[2] = xtest[2] * (14.0 - 4.0) + 4.0
        xtest[3] = xtest[3] * (16.0 * np.pi / 180.0) - 8.0 * np.pi / 180.0
        xtest[4] = xtest[4] * (24.0 * np.pi / 180.0) - 4.0 * np.pi / 180.0
        xtest[5] = xtest[5] * (36.0 * np.pi / 180.0) - 18.0 * np.pi / 180.0

        if self.collision_free(np.array([xtest])):
            return xtest
        else:
            return self.sample_point()

    def reconstruct_path(self):
        current = self.nodes[-1]

        utraj, xtraj, ttraj, res = self.plant.trajOptRRT(current.parent.state, current.state, goal=True)
        urrt = utraj
        xrrt = xtraj
        trrt = ttraj
        current = current.parent
        while current.parent is not None:
            utraj, xtraj, ttraj, res = self.plant.trajOptRRT(current.parent.state, current.state, goal=False)
            urrt = np.vstack((utraj, urrt))
            xrrt = np.vstack((xtraj[:-1,:], xrrt))
            trrt = np.hstack((ttraj[:-1], trrt+ttraj.max()))
            current = current.parent

        # save traj
        # TODO: do something else - this is sloppy
        self.plant.udtraj = urrt
        self.plant.xdtraj = xrrt
        self.plant.ttraj = trrt
        self.plant.mp_result = res

        self.plant.udtraj_poly = PiecewisePolynomial.FirstOrderHold(trrt[0:-1], urrt.T)
        self.plant.xdtraj_poly = PiecewisePolynomial.Cubic(trrt, xrrt.T)

        return urrt, xrrt, trrt


class Node(object):
    def __init__(self, state, parent_node=None):
        self.parent = parent_node
        self.state = state
