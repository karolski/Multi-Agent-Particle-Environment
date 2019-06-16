import numpy as np

# physical/external base state of all entites
class EntityState(object):
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity
        self.p_vel = None

# state of agents (including communication and internal/mental state)
class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        # communication utterance
        self.c = None

# action of the agent
class Action(object):
    def __init__(self):
        # physical action
        self.u = None
        # communication action
        self.c = None

# properties and state of physical world entity
class Entity(object):
    def __init__(self):
        # name 
        self.name = ''
        # properties:
        self.size = 0.050
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = None
        # max speed and accel
        self.max_speed = None
        self.accel = None
        # state
        self.state = EntityState()
        # mass
        self.initial_mass = 1.0
        # shape, defaults to circle,
        self.shape = None
        # Coordinates of points in a polygon
        self.polygon_shape=None

    @property
    def mass(self):
        return self.initial_mass

# properties of landmark entities
class Landmark(Entity):
     def __init__(self):
        super(Landmark, self).__init__()

# properties of agent entities
class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = False
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = None
        # communication noise amount
        self.c_noise = None
        # control range
        self.u_range = 1.0
        # state
        self.state = AgentState()
        # action
        self.action = Action()
        # script behavior to execute
        self.action_callback = None

# multi-agent world
class World(object):
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 2
        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        self.dt = 0.1
        # physical damping
        self.damping = 0.25
        # contact response parameters
        self.contact_force = 1e+2
        self.contact_margin = 1e-3

    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.landmarks

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    # update state of the world
    def step(self):
        # set actions for scripted agents 
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)
        # gather forces applied to entities
        p_force = [None] * len(self.entities)
        # apply agent physical controls
        p_force = self.apply_action_force(p_force)
        # apply environment forces
        p_force = self.apply_environment_force(p_force)
        # integrate physical state
        self.integrate_state(p_force)
        # update agent state
        for agent in self.agents:
            self.update_agent_state(agent)

    # gather agent action forces
    def apply_action_force(self, p_force):
        # set applied forces
        for i,agent in enumerate(self.agents):
            if agent.movable:
                noise = np.random.randn(*agent.action.u.shape) * agent.u_noise if agent.u_noise else 0.0
                p_force[i] = agent.action.u + noise                
        return p_force

    # gather physical forces acting on entities
    def apply_environment_force(self, p_force):
        # simple (but inefficient) collision response
        for a,entity_a in enumerate(self.entities):
            for b,entity_b in enumerate(self.entities):
                if(b <= a): continue
                [f_a, f_b] = self.get_collision_force(entity_a, entity_b)
                if(f_a is not None):
                    if(p_force[a] is None): p_force[a] = 0.0
                    p_force[a] = f_a + p_force[a] 
                if(f_b is not None):
                    if(p_force[b] is None): p_force[b] = 0.0
                    p_force[b] = f_b + p_force[b]        
        return p_force

    # integrate physical state
    def integrate_state(self, p_force):
        for i,entity in enumerate(self.entities):
            if not entity.movable: continue
            entity.state.p_vel = entity.state.p_vel * (1 - self.damping)
            if (p_force[i] is not None):
                entity.state.p_vel += (p_force[i] / entity.mass) * self.dt
            if entity.max_speed is not None:
                speed = np.sqrt(np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1]))
                if speed > entity.max_speed:
                    entity.state.p_vel = entity.state.p_vel / np.sqrt(np.square(entity.state.p_vel[0]) +
                                                                  np.square(entity.state.p_vel[1])) * entity.max_speed
            entity.state.p_pos += entity.state.p_vel * self.dt

    def update_agent_state(self, agent):
        # set communication state (directly for now)
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = np.random.randn(*agent.action.c.shape) * agent.c_noise if agent.c_noise else 0.0
            agent.state.c = agent.action.c + noise

    def get_dist_from_boundries(self, entity_a):
        boundry_horizontal = 1
        boundry_vertical = 1
        if entity_a.state.p_pos[0] > 0:
            delta_x = entity_a.state.p_pos[0] - boundry_horizontal
            if delta_x > 0: delta_x = - 0.5 * delta_x

        else:
            delta_x = entity_a.state.p_pos[0] + boundry_horizontal
            if delta_x < 0: delta_x = - 0.5 * delta_x #if entity went beyond boundries, reverse signs

        if entity_a.state.p_pos[1] > 0:
            delta_y = entity_a.state.p_pos[1] - boundry_vertical
            if delta_y > 0: delta_y = - 0.5 * delta_y #if entity went beyond boundries, reverse signs
        else:
            delta_y = entity_a.state.p_pos[1] + boundry_vertical
            if delta_y < 0: delta_y = - 0.5 * delta_y #if entity went beyond boundries, reverse signs



        if abs(delta_x) < abs(delta_y):
            delta_pos = np.array([delta_x, 0])
            dist = abs(delta_x)
        else:
            delta_pos = np.array([0, delta_y])
            dist = abs(delta_y)
        dist_min = entity_a.size + 0.15

        return dist, dist_min, delta_pos

    def get_dist_from_square_entity(self, entity_a, square_entity):

        left_wall = square_entity.state.p_pos[0]
        right_wall = square_entity.state.p_pos[0] + square_entity.size
        upper_wall = square_entity.state.p_pos[1] + square_entity.size
        lower_wall = square_entity.state.p_pos[1]
        delta_x = None
        delta_y = None
        dist = 2e2
        delta_pos = np.array([0,0])
        dist_min = entity_a.size

        if entity_a.state.p_pos[0] > right_wall:
            delta_x = entity_a.state.p_pos[0]- right_wall
        elif entity_a.state.p_pos[0] < left_wall:
            delta_x = entity_a.state.p_pos[0] - left_wall
        if entity_a.state.p_pos[1] > upper_wall:
            delta_y = entity_a.state.p_pos[1] - upper_wall
        elif entity_a.state.p_pos[1] < lower_wall:
            delta_y = entity_a.state.p_pos[1] - lower_wall
        else: #when the entity got inside
            distances = {
                "to_upper_wall": upper_wall - entity_a.state.p_pos[1] ,
                "to_lower_wall": entity_a.state.p_pos[1] - lower_wall,
                "to_left_wall": entity_a.state.p_pos[0] - left_wall,
                "to_right_wall": right_wall - entity_a.state.p_pos[0]
            }
            delta_pos_to_oposite_wall = {
                "to_upper_wall": np.array([0,distances["to_lower_wall"]]),
                "to_lower_wall": np.array([0,-distances["to_upper_wall"]]),
                "to_left_wall": np.array([-distances["to_right_wall"],0]),
                "to_right_wall": np.array([distances["to_left_wall"],0])
            }
            dist = square_entity.size - min(distances.values())
            closest_wall = min(distances, key=distances.get)
            delta_pos = delta_pos_to_oposite_wall[closest_wall]
            dist_min = square_entity.size + entity_a.size
            return dist, dist_min, delta_pos

        if delta_y and delta_x:
            #possibly hitting the corner
            delta_pos = np.array([delta_y, delta_x])
            dist = np.mean( [abs(delta_y), abs(delta_x)])

        elif delta_x:
            if(delta_x) <  entity_a.size and delta_y is None:
                #Hitting only up or down
                delta_pos = np.array([delta_x, 0])
                dist = abs(delta_x)
        elif delta_y:
            if abs(delta_y) <  entity_a.size and delta_x is None:
                #hitting only left or right
                delta_pos = np.array([0, delta_y])
                dist = abs(delta_y)


        #correct for when entity gets inside the square
        # dist += entity_a.size
        # dist_min += entity_a.size
        # delta_pos = 2*delta_pos
        return dist, dist_min, delta_pos


    # get collision forces for any contact between two entities
    def get_collision_force(self, entity_a, entity_b):
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None] # not a collider
        if (entity_a is entity_b):
            return [None, None] # don't collide against itself
        if (not entity_a.movable) and (not entity_b.movable):
            return [None, None]

        # compute actual distance between entities or landmark

        if entity_b.name == "land":
            dist, dist_min, delta_pos = self.get_dist_from_boundries(entity_a)
        elif entity_b.shape == "square":
            dist, dist_min, delta_pos = self.get_dist_from_square_entity(entity_a, entity_b)

        else:
            delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
            dist = np.sqrt(np.sum(np.square(delta_pos)))
            # minimum allowable distance
            dist_min = entity_a.size + entity_b.size


        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min)/k)*k
        force = self.contact_force * delta_pos / dist * penetration
        force_a = +force if entity_a.movable else None
        force_b = -force if entity_b.movable else None
        return [force_a, force_b]