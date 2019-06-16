import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def __init__(self):
        self.sand_color_array = np.array([0.75, 0.7, 0.5])
        self.sand_circle_positions = []
        self.sand_circle_sizes = []


    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 0
        num_agents = 3
        num_landmarks = 4 #Landmarks are land and a berth
        world.collaborative = False
        world.damping = 0.5
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.10
            agent.docked = False


        # add landmarks
        self.land = Landmark()
        self.land.name = "land"
        self.land.shape = "land"

        self.berth = Landmark()
        self.berth.name = "berth"
        self.berth.collide = False
        self.berth.movable = False
        self.berth.shape = "box"
        self.berth.size = 0.5
        self.berth.color = np.array([0.1, 0.5, 0.1])

        self.exit = Landmark()
        self.exit.name = "exit"
        self.exit.collide = False
        self.exit.movable = False
        self.exit.shape = "box"
        self.exit.size = 0.5
        self.exit.color = np.array([0.4, 0.1, 0.1])


        self.upper_ball = Landmark()
        self.upper_ball.name = "upper_ball"
        self.upper_ball.size = 0.4

        self.lower_ball = Landmark()
        self.lower_ball.size = 0.4
        self.lower_ball.name = "lower_ball"

        self.square = Landmark()
        self.square.size = 0.8
        self.square.name = "square"
        self.square.shape = "square"
        self.upper_square = Landmark()
        self.upper_square.size = 1
        self.upper_square.name = "upper square"
        self.upper_square.shape = "square"

        self.berth.state.p_pos = np.array([0.5, -0.7])
        self.exit.state.p_pos = np.array([-1, -0.7])
        self.land.state.p_pos = np.array([-1, -0.925])
        self.lower_ball.state.p_pos = np.array([-0.1, -0.5])
        self.upper_ball.state.p_pos = np.array([0, 0.5])
        self.square.state.p_pos = np.array([-0.5,-1.2])
        self.upper_square.state.p_pos = np.array([-0.5, 0.4])

        world.landmarks = [self.upper_ball,
                           self.lower_ball,
                           self.berth,
                           self.exit,
                           self.land,
                           self.square,
                           self.upper_square]

        self.land_landmarks = [self.upper_ball,
                               self.lower_ball,
                               self.land,
                               self.square,
                               self.upper_square]

        for l in self.land_landmarks:
            l.color = self.sand_color_array
            l.collide = True
            l.movable = False
            l.state.p_vel = np.zeros(world.dim_p)



        self.reset_world(world)


        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])

        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            agent.in_harbour = True
            # agent.docked = False








    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)


    def is_collision(self, agent1, agent2):
        if agent1 == agent2:
            return False
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def is_at_any_collision(self, agent, world):
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    return True
        return False

    def is_at_berth(self, agent):
        berth_x = self.berth.state.p_pos[0]
        berth_y = self.berth.state.p_pos[1]

        is_aligned_with_berth_on_x = berth_x < agent.state.p_pos[0] < (berth_x + self.berth.size)
        is_aligned_with_berth_on_y = berth_y < agent.state.p_pos[1] < (berth_y + self.berth.size)
        is_at_berth = is_aligned_with_berth_on_x and is_aligned_with_berth_on_y
        return is_at_berth

    def is_at_exit(self, agent):
        is_aligned_with_exit_on_x = abs(agent.state.p_pos[0] - self.exit.state.p_pos[0]) < self.exit.size
        is_aligned_with_exit_on_y = abs(agent.state.p_pos[1] - self.exit.state.p_pos[1]) < self.exit.size
        is_at_exit = (is_aligned_with_exit_on_x and is_aligned_with_exit_on_y)
        return is_at_exit

    def reward(self, agent, world):
        FUEL_COEFFICIENT = 0.001
        TIME_CONSTANT = 0.001
        COLLISION_PUNISHMENT = 0.1

        rew = 0
        if not agent.in_harbour:
            return 0
        velocity = np.sqrt(np.sum(np.square(agent.state.p_vel)))
        rew -= velocity*FUEL_COEFFICIENT
        rew -= TIME_CONSTANT
        rew -= float(self.is_at_any_collision(agent,world)) * COLLISION_PUNISHMENT
        if not agent.docked:
            if self.is_at_berth(agent):
                rew += 5
                agent.docked = True
        else:
            if self.is_at_exit(agent):
                rew += 5
                agent.docked = False
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        other_pos = []
        other_vel = []
        other_docked = []
        for other in world.agents:
            if other is agent: continue
            other_pos.append(other.state.p_pos) #modified from difference with position of the agent to independent position
            other_vel.append(other.state.p_vel)
            other_docked.append([float(other.docked)])
        return np.concatenate([[float(agent.docked)]]+ [agent.state.p_vel] + [agent.state.p_pos] + other_pos + other_vel + other_docked)
