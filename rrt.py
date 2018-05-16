from pydrake.solvers._mathematicalprogram_py import SolutionResult
from pydrake.trajectories import PiecewisePolynomial
import numpy as np
import time


class RRTStar(object):
    def __init__(self, x0, xG, plant, obs):
        self.x0 = x0         # initial state
        self.xG = xG         # goal state
        self.plant = plant   # AirplaneSystem()
        self.obs = obs       # array of obstacles
        self.nodes = list()  # list of nodes in tree
        self.best_goal_node = None  # track the best goal node
        self.counts = {'x_near' : 0, 'x_near_free' : 0}  # some counters for analysis

    def run(self):
        """
        This is the main method of RRTStar(), which carries out a sampling-based motion planning and obstacle
         avoidance algorithm roughly based on RRT*.
        """
        t_start = time.time()

        # number of iterations
        N = 200

        # root node
        start_node = Node(self.x0)

        # add root node to list of all nodes in tree
        self.nodes.append(start_node)

        # test whether goal is reachable and obstacle free from start point
        utraj, xtraj, ttraj, res, cost = self.plant.trajOptRRT(self.x0, self.xG, goal=True)

        if res == SolutionResult.kSolutionFound:
            if self.collision_free(xtraj):
                print "goal reachable from start and obstacle free"
                goal_node = Node(self.xG, parent_node=start_node, cost=cost, goal_node=True)
                self.nodes.append(goal_node)
                self.best_goal_node = goal_node
                return self.best_goal_node
            else:
                print "goal reachable from start but not obstacle free"
        else:
            print "goal not reachable from start"

        for i in range(N):
            print "i: ", i

            # get a random point in bounds that doesn't intersect with obstacles
            x_rand = self.sample_point()

            # find nearest existing node in tree and also lists of backward/forward nodes
            near_node, backward_nodes, forward_nodes = self.nearest(x_rand)

            if near_node is not None:
                x_near = near_node.state

                # keep track of number of random points being tested for connection
                self.counts['x_near'] += 1

                # check for dynamically feasible path
                utraj, xtraj, ttraj, res, cost = self.plant.trajOptRRT(x_near, x_rand, goal=False)

                if res == SolutionResult.kSolutionFound:
                    if self.collision_free(xtraj):
                        # connection found between x_near and x_rand
                        new_node = Node(x_rand, parent_node=near_node, cost=cost, goal_node=False)

                        # track number of points which will be added to tree
                        self.counts['x_near_free'] += 1

                        print 'backwards nodes: ', len(backward_nodes), ', forwards nodes: ', len(forward_nodes)

                        # check for a better node to go through than near_node
                        for node in backward_nodes:
                            # don't look at the node already connected to
                            if node.state is not near_node.state:
                                utraj, xtraj, ttraj, res, cost = self.plant.trajOptRRT(node.state, x_rand, goal=False)
                                if res == SolutionResult.kSolutionFound:
                                    if self.collision_free(xtraj):
                                        # another feasible connection found, so check cost

                                        new_cost = self.sum_cost(node) + cost
                                        if new_cost < self.sum_cost(new_node):
                                            # connect through this node
                                            print 'better path to new node found'
                                            new_node.cost = cost
                                            new_node.parent = node

                        # look at rewiring other nodes in the tree (forward_nodes) through new node
                        for node in forward_nodes:
                            # don't look at the node already connected to
                            if node.state is not near_node.state:
                                utraj, xtraj, ttraj, res, cost = self.plant.trajOptRRT(new_node.state, node.state, goal=node.goal_node)
                                if res == SolutionResult.kSolutionFound:
                                    if self.collision_free(xtraj):
                                        # another feasible connection found

                                        new_cost = self.sum_cost(new_node) + cost
                                        if new_cost < self.sum_cost(node):
                                            print 'better path through new node found'
                                            replacement_node = Node(node.state, parent_node=new_node, cost=cost, goal_node=node.goal_node)
                                            self.nodes.remove(node)
                                            self.nodes.append(replacement_node)
                                            # if node is a goal node, check whether new cost to it is lowest
                                            if replacement_node.goal_node and new_cost < self.sum_cost(self.best_goal_node):
                                                print 'better RRT* path to goal found (1)'
                                                self.best_goal_node = replacement_node

                        # finally, add this node to the tree
                        self.nodes.append(new_node)

                        # check whether this node can reach goal
                        utraj, xtraj, ttraj, res, cost = self.plant.trajOptRRT(x_rand, self.xG, goal=True)
                        if res == SolutionResult.kSolutionFound:
                            if self.collision_free(xtraj):
                                goal_node = Node(self.xG, parent_node=new_node, cost=cost, goal_node=True)
                                if self.best_goal_node is None:
                                    self.best_goal_node = goal_node
                                    print 'first RRT* path to goal found'
                                elif (self.sum_cost(new_node) + cost) < self.sum_cost(self.best_goal_node):
                                    self.best_goal_node = goal_node
                                    print 'better RRT* path to goal found (2)'
                                self.nodes.append(goal_node)

        t_finish = time.time()
        t_diff = t_finish - t_start
        print('RRT* execution time: %.3f seconds' % t_diff)
        return self.best_goal_node

    def collision_free(self, xtraj):
        """
        Check whether any of the (x,z) knot points along trajectory collide with obstacles
        """
        for xt in xtraj:
            for ob in self.obs:
                if xt[0] > ob[0] and xt[0] < (ob[0] + ob[2]) and xt[1] > ob[1] and xt[1] < (ob[1] + ob[3]):
                    return False
        return True

    def nearest(self, test_pt):
        """
        Find nearest node to test_pt, roughly based on RRT*. Lists of backward and forward nodes are returned,
        for rewiring steps of RRT*. Backward nodes are checked to find a shorter path to new node (test_pt), and
        forward nodes are checked to rewire through new node (test_pt).
        Nearest node is computed using discrete probability distribution on list of backward nodes sorted by
        x-component of distance.
        """
        dists = []
        backward_nodes = []
        forward_nodes = []

        for node in self.nodes:
            if node.state[0] < (test_pt[0] - 3.0):
                backward_nodes.append(node)
                dists.append((node.state[0] - test_pt[0]) ** 2 + (node.state[1] - test_pt[1]) ** 2)

            elif node.state[0] > (test_pt[0] + 3.0):
                forward_nodes.append(node)

        sorted_inds = np.argsort(dists)

        if len(backward_nodes) == 0:
            return self.nodes[0], backward_nodes, forward_nodes
        else:
            p = 3.0 / len(backward_nodes)
            p = min((p, 0.8))
            flip = np.random.binomial(1, p)
            ind = 0
            while flip == 0:
                flip = np.random.binomial(1, p)
                if ind < len(backward_nodes) - 1:
                    ind += 1
                else:
                    ind = 0
            return backward_nodes[sorted_inds[ind]], backward_nodes, forward_nodes

    def sample_point(self):
        """
        Generate a new point to sample, within reasonable state bounds and relatively close to a trim condition.
        Metric of closeness to trim uses total magnitude of accelerations at xtest w/zero input.
        """
        xtest = np.random.rand(self.plant.num_states)  # all uniform 0-1
        xtest[0] = xtest[0] * (self.xG[0] - self.x0[0]) + self.x0[0]
        xtest[1] = xtest[1] * (2.0 - 0.3) + 0.3
        xtest[2] = xtest[2] * (14.0 - 4.0) + 4.0
        xtest[3] = xtest[3] * (16.0 * np.pi / 180.0) - 8.0 * np.pi / 180.0
        xtest[4] = xtest[4] * (24.0 * np.pi / 180.0) - 4.0 * np.pi / 180.0
        xtest[5] = xtest[5] * (36.0 * np.pi / 180.0) - 18.0 * np.pi / 180.0

        if self.collision_free(np.array([xtest])):
            f = self.plant.airplaneLongDynamics(xtest, np.zeros(2))
            fmag = 0.0
            for fi in f[[2, 3, 5]]:
                fmag += np.abs(fi)
            fmag = max((1, fmag))
            good_pt = np.random.binomial(1, 1/fmag, 1)
            if good_pt:
                return xtest
            else:
                return self.sample_point()
        else:
            return self.sample_point()

    def sum_cost(self, node):
        """
        Add up the costs along the path to node.
        """
        total_cost = 0.0
        while node.parent is not None:
            assert(node.cost >= 0.0)
            total_cost += node.cost
            node = node.parent
        return total_cost

    def reconstruct_path(self):
        """
        Piece together trajectories between nodes in the path to goal.
        """
        current = self.best_goal_node

        utraj, xtraj, ttraj, res, cost= self.plant.trajOptRRT(current.parent.state, current.state, goal=True)
        urrt = utraj
        xrrt = xtraj
        trrt = ttraj
        current = current.parent
        while current.parent is not None:
            utraj, xtraj, ttraj, res, cost = self.plant.trajOptRRT(current.parent.state, current.state, goal=False)
            urrt = np.vstack((utraj, urrt))
            xrrt = np.vstack((xtraj[:-1, :], xrrt))
            trrt = np.hstack((ttraj[:-1], trrt + ttraj.max()))
            current = current.parent

        # save traj
        # TODO: do something else - this is sloppy
        self.plant.udtraj = urrt
        self.plant.xdtraj = xrrt
        self.plant.ttraj = trrt
        self.plant.mp_result = res

        self.plant.udtraj_poly = PiecewisePolynomial.FirstOrderHold(trrt[0:-1], urrt.T)
        self.plant.xdtraj_poly = PiecewisePolynomial.Cubic(trrt, xrrt.T)

        print '\n', 'total cost of RRT* path: ', self.sum_cost(self.best_goal_node)
        return urrt, xrrt, trrt


class Node(object):
    """
    Node() holds information on RRT* search nodes
    """
    def __init__(self, state, parent_node=None, cost=None, goal_node=False):
        self.parent = parent_node
        self.state = state
        self.cost = cost
        self.goal_node = goal_node
