"""
Rocket trajectory optimization is a classic topic in Optimal Control.
According to Pontryagin's maximum principle it's optimal to fire engine full throttle or
turn it off. That's the reason this environment is OK to have discreet actions (engine on or off).
The landing pad is always at coordinates (0,0). The coordinates are the first two numbers in the state vector.
Reward for moving from the top of the screen to the landing pad and zero speed is about 100..140 points.
If the lander moves away from the landing pad it loses reward. The episode finishes if the lander crashes or
comes to rest, receiving an additional -100 or +100 points. Each leg with ground contact is +10 points.
Firing the main engine is -0.3 points each frame. Firing the side engine is -0.03 points each frame.
Solved is 200 points.
Landing outside the landing pad is possible. Fuel is infinite, so an agent can learn to fly and then land
on its first attempt. Please see the source code for details.
To see a heuristic landing, run:
python gym/envs/box2d/lunar_lander.py
To play yourself, run:
python examples/agents/keyboard_agent.py LunarLander-v2
Created by Oleg Klimov. Licensed on the same terms as the rest of OpenAI Gym.
"""


import sys, math
import numpy as np

import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)

import gym
from gym import spaces
from gym.utils import seeding, EzPickle

FPS = 50
SCALE = 30.0   # affects how fast-paced the game is, forces should be adjusted as well

MAIN_ENGINE_POWER = 13.0
SIDE_ENGINE_POWER = 0.6

INITIAL_RANDOM = 1000.0   # Set 1500 to make game harder

LANDER_POLY =[
    (-14, +17), (-17, 0), (-17 ,-10),
    (+17, -10), (+17, 0), (+14, +17)
    ]
LEG_AWAY = 20
LEG_DOWN = 18
LEG_W, LEG_H = 2, 8
LEG_SPRING_TORQUE = 40

SIDE_ENGINE_HEIGHT = 14.0
SIDE_ENGINE_AWAY = 12.0

VIEWPORT_W = 600
VIEWPORT_H = 400


class ContactDetector(contactListener):
    def __init__(self, env, n=0):
        contactListener.__init__(self)
        self.env = env
        self.n = n

    def BeginContact(self, contact):
        if self.env.landers[self.n] == contact.fixtureA.body or self.env.landers[self.n] == contact.fixtureB.body:
            self.env.game_overs[self.n] = True
        for i in range(2):
            if self.env.all_legs[self.n][i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.all_legs[self.n][i].ground_contact = True

    def EndContact(self, contact):
        for i in range(2):
            if self.env.all_legs[self.n][i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.all_legs[self.n][i].ground_contact = False


class LunarLanderMulti(gym.Env, EzPickle):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : FPS
    }

    continuous = False

    def __init__(self, N=1):
        EzPickle.__init__(self)
        self.seed()
        self.viewer = None

        self.worlds = [Box2D.b2World() for n in range(N)]
        self.moons = [None] * N
        self.landers = [None] * N
        self.all_particles = [[] for _ in range(N)]

        self.N = N
        self.prev_reward = None

        # useful range is -1 .. +1, but spikes can be higher
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32)

        if self.continuous:
            # Action is two floats [main engine, left-right engines].
            # Main engine: -1..0 off, 0..+1 throttle from 50% to 100% power. Engine can't work with less than 50% power.
            # Left-right:  -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off
            self.action_space = spaces.Box(-1, +1, (2,), dtype=np.float32)
        else:
            # Nop, fire left engine, main engine, right engine
            self.action_space = spaces.Discrete(4)

        self.fuels = [None] * N  # Custom

        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self, n=0):
        if not self.moons[n]: return
        self.worlds[n].contactListener = None
        self._clean_particles(True, n)
        self.worlds[n].DestroyBody(self.moons[n])
        self.moons[n] = None
        self.worlds[n].DestroyBody(self.landers[n])
        self.landers[n] = None
        self.worlds[n].DestroyBody(self.all_legs[n][0])
        self.worlds[n].DestroyBody(self.all_legs[n][1])

    def reset(self):

        for n in range(self.N):
            self._destroy(n)
            self.worlds[n].contactListener_keepref = ContactDetector(self, n)
            self.worlds[n].contactListener = self.worlds[n].contactListener_keepref

        self.game_overs = [False] * self.N
        self.prev_shapings = [None] * self.N

        W = VIEWPORT_W/SCALE
        H = VIEWPORT_H/SCALE

        # terrain
        CHUNKS = 11
        height = self.np_random.uniform(0, H/2, size=(CHUNKS+1,))
        chunk_x = [W/(CHUNKS-1)*i for i in range(CHUNKS)]
        self.helipad_x1 = chunk_x[CHUNKS//2-1]
        self.helipad_x2 = chunk_x[CHUNKS//2+1]
        self.helipad_y = H/4
        height[CHUNKS//2-2] = self.helipad_y
        height[CHUNKS//2-1] = self.helipad_y
        height[CHUNKS//2+0] = self.helipad_y
        height[CHUNKS//2+1] = self.helipad_y
        height[CHUNKS//2+2] = self.helipad_y
        smooth_y = [0.33*(height[i-1] + height[i+0] + height[i+1]) for i in range(CHUNKS)]

        self.moons = [None] * self.N
        for n in range(self.N):
            moon = self.worlds[n].CreateStaticBody(shapes=edgeShape(vertices=[(0, 0), (W, 0)]))

            for i in range(CHUNKS-1):
                p1 = (chunk_x[i], smooth_y[i])
                p2 = (chunk_x[i+1], smooth_y[i+1])
                moon.CreateEdgeFixture(
                    vertices=[p1,p2],
                    density=0,
                    friction=0.1)
            moon.color1 = (0.0, 0.0, 0.0)
            moon.color2 = (0.0, 0.0, 0.0)
            self.moons[n] = moon

        self.sky_polys = []
        for i in range(CHUNKS-1):
            p1 = (chunk_x[i], smooth_y[i])
            p2 = (chunk_x[i + 1], smooth_y[i + 1])
            self.sky_polys.append([p1, p2, (p2[0], H), (p1[0], H)])

        initial_y = VIEWPORT_H/SCALE

        rd_v_x = self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM)
        rd_v_y = self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM)

        self.landers = [None] * self.N  # List of different landers
        self.all_legs = [None] * self.N
        for n in range(self.N):
            self.landers[n] = self.worlds[n].CreateDynamicBody(
                position=(VIEWPORT_W/SCALE/2, initial_y),
                angle=0.0,
                fixtures = fixtureDef(
                    shape=polygonShape(vertices=[(x/SCALE, y/SCALE) for x, y in LANDER_POLY]),
                    density=5.0,
                    friction=0.1,
                    categoryBits=0x0010,
                    maskBits=0x001,   # collide only with ground
                    restitution=0.0)  # 0.99 bouncy
                    )
            self.landers[n].color1 = (0.5, 0.4, 0.9)
            self.landers[n].color2 = (0.3, 0.3, 0.5)
            self.landers[n].ApplyForceToCenter( (
                rd_v_x,
                rd_v_y
                ), True)

            legs = []
            for i in [-1, +1]:
                leg = self.worlds[n].CreateDynamicBody(
                    position=(VIEWPORT_W/SCALE/2 - i*LEG_AWAY/SCALE, initial_y),
                    angle=(i * 0.05),
                    fixtures=fixtureDef(
                        shape=polygonShape(box=(LEG_W/SCALE, LEG_H/SCALE)),
                        density=1.0,
                        restitution=0.0,
                        categoryBits=0x0020,
                        maskBits=0x001)
                    )
                leg.ground_contact = False
                leg.color1 = (0.5, 0.4, 0.9)
                leg.color2 = (0.3, 0.3, 0.5)
                rjd = revoluteJointDef(
                    bodyA=self.landers[n],
                    bodyB=leg,
                    localAnchorA=(0, 0),
                    localAnchorB=(i * LEG_AWAY/SCALE, LEG_DOWN/SCALE),
                    enableMotor=True,
                    enableLimit=True,
                    maxMotorTorque=LEG_SPRING_TORQUE,
                    motorSpeed=+0.3 * i  # low enough not to jump back into the sky
                    )
                if i == -1:
                    rjd.lowerAngle = +0.9 - 0.5  # The most esoteric numbers here, angled legs have freedom to travel within
                    rjd.upperAngle = +0.9
                else:
                    rjd.lowerAngle = -0.9
                    rjd.upperAngle = -0.9 + 0.5
                leg.joint = self.worlds[n].CreateJoint(rjd)
                legs.append(leg)
            self.all_legs[n] = legs

        self.drawlist = self.landers

        for legs in self.all_legs:
            self.drawlist += legs

        self.fuels = [0.] * self.N  # Custom

        return self.step(np.array([0, 0]) if self.continuous else 0)[0]

    def _create_particle(self, mass, x, y, ttl, n=0):

        p = self.worlds[n].CreateDynamicBody(
            position = (x, y),
            angle=0.0,
            fixtures = fixtureDef(
                shape=circleShape(radius=2/SCALE, pos=(0, 0)),
                density=mass,
                friction=0.1,
                categoryBits=0x0100,
                maskBits=0x001,  # collide only with ground
                restitution=0.3)
                )
        p.ttl = ttl
        self.all_particles[n].append(p)
        self._clean_particles(False, n)
        return p

    def _clean_particles(self, all, n=0):
        while self.all_particles[n] and (all or self.all_particles[n][0].ttl < 0):
            self.worlds[n].DestroyBody(self.all_particles[n].pop(0))

    def step(self, action, n=0):  # n = agent to move
        if self.game_overs[n]:
            return None, None, True, {}

        curr_lander = self.landers[n]

        if self.continuous:
            action = np.clip(action, -1, +1).astype(np.float32)
        else:
            assert self.action_space.contains(action), "%r (%s) invalid " % (action, type(action))

        # Engines
        tip = (math.sin(curr_lander.angle), math.cos(curr_lander.angle))
        side = (-tip[1], tip[0])
        dispersion = [self.np_random.uniform(-1.0, +1.0) / SCALE for _ in range(2)]

        m_power = 0.0
        if (self.continuous and action[0] > 0.0) or (not self.continuous and action == 2):
            # Main engine
            if self.continuous:
                m_power = (np.clip(action[0], 0.0,1.0) + 1.0)*0.5   # 0.5..1.0
                assert m_power >= 0.5 and m_power <= 1.0
            else:
                m_power = 1.0
            ox = (tip[0] * (4/SCALE + 2 * dispersion[0]) +
                  side[0] * dispersion[1])  # 4 is move a bit downwards, +-2 for randomness
            oy = -tip[1] * (4/SCALE + 2 * dispersion[0]) - side[1] * dispersion[1]
            impulse_pos = (curr_lander.position[0] + ox, curr_lander.position[1] + oy)
            p = self._create_particle(3.5,  # 3.5 is here to make particle speed adequate
                                      impulse_pos[0],
                                      impulse_pos[1],
                                      m_power,
                                      n)  # particles are just a decoration
            p.ApplyLinearImpulse((ox * MAIN_ENGINE_POWER * m_power, oy * MAIN_ENGINE_POWER * m_power),
                                 impulse_pos,
                                 True)
            curr_lander.ApplyLinearImpulse((-ox * MAIN_ENGINE_POWER * m_power, -oy * MAIN_ENGINE_POWER * m_power),
                                           impulse_pos,
                                           True)

        s_power = 0.0
        if (self.continuous and np.abs(action[1]) > 0.5) or (not self.continuous and action in [1, 3]):
            # Orientation engines
            if self.continuous:
                direction = np.sign(action[1])
                s_power = np.clip(np.abs(action[1]), 0.5, 1.0)
                assert s_power >= 0.5 and s_power <= 1.0
            else:
                direction = action-2
                s_power = 1.0
            ox = tip[0] * dispersion[0] + side[0] * (3 * dispersion[1] + direction * SIDE_ENGINE_AWAY/SCALE)
            oy = -tip[1] * dispersion[0] - side[1] * (3 * dispersion[1] + direction * SIDE_ENGINE_AWAY/SCALE)
            impulse_pos = (curr_lander.position[0] + ox - tip[0] * 17/SCALE,
                           curr_lander.position[1] + oy + tip[1] * SIDE_ENGINE_HEIGHT/SCALE)
            p = self._create_particle(0.7, impulse_pos[0], impulse_pos[1], s_power, n)
            p.ApplyLinearImpulse((ox * SIDE_ENGINE_POWER * s_power, oy * SIDE_ENGINE_POWER * s_power),
                                 impulse_pos
                                 , True)
            curr_lander.ApplyLinearImpulse((-ox * SIDE_ENGINE_POWER * s_power, -oy * SIDE_ENGINE_POWER * s_power),
                                           impulse_pos,
                                           True)

        self.worlds[n].Step(1.0/FPS, 6*30, 2*30)

        pos = curr_lander.position
        vel = curr_lander.linearVelocity
        state = [
            (pos.x - VIEWPORT_W/SCALE/2) / (VIEWPORT_W/SCALE/2),
            (pos.y - (self.helipad_y+LEG_DOWN/SCALE)) / (VIEWPORT_H/SCALE/2),
            vel.x*(VIEWPORT_W/SCALE/2)/FPS,
            vel.y*(VIEWPORT_H/SCALE/2)/FPS,
            curr_lander.angle,
            20.0*curr_lander.angularVelocity/FPS,
            1.0 if self.all_legs[n][0].ground_contact else 0.0,
            1.0 if self.all_legs[n][1].ground_contact else 0.0
            ]
        assert len(state) == 8

        reward = 0
        shaping = \
            - 100*np.sqrt(state[0]*state[0] + state[1]*state[1]) \
            - 100*np.sqrt(state[2]*state[2] + state[3]*state[3]) \
            - 100*abs(state[4]) + 10*state[6] + 10*state[7]  # And ten points for legs contact, the idea is if you
                                                             # lose contact again after landing, you get negative reward
        if self.prev_shapings[n] is not None:
            reward = shaping - self.prev_shapings[n]
        self.prev_shapings[n] = shaping

        reward -= m_power*0.30  # less fuel spent is better, about -30 for heuristic landing
        reward -= s_power*0.03

        self.fuels[n] += m_power*0.30 + s_power*0.03

        game_state = "not_ended"
        done = False
        if self.game_overs[n] or abs(state[0]) >= 1.0:
            done = True
            reward = -100
            game_state = "lost"
        if not curr_lander.awake:
            done = True
            reward = +100
            game_state = "won"

        return np.array(state, dtype=np.float32), reward, done, {'game_state': game_state, 'fuel': self.fuels[n]}

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
            self.viewer.set_bounds(0, VIEWPORT_W/SCALE, 0, VIEWPORT_H/SCALE)

        for n in range(self.N):
            for obj in self.all_particles[n]:
                obj.ttl -= 0.15
                obj.color1 = (max(0.2, 0.2+obj.ttl), max(0.2, 0.5*obj.ttl), max(0.2, 0.5*obj.ttl))
                obj.color2 = (max(0.2, 0.2+obj.ttl), max(0.2, 0.5*obj.ttl), max(0.2, 0.5*obj.ttl))

            self._clean_particles(False, n)

        for p in self.sky_polys:
            self.viewer.draw_polygon(p, color=(0, 0, 0))

        particles = []
        for n in range(self.N):
            particles += self.all_particles[n]

        for obj in particles + self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    t = rendering.Transform(translation=trans*f.shape.pos)
                    self.viewer.draw_circle(f.shape.radius, 20, color=obj.color1).add_attr(t)
                    self.viewer.draw_circle(f.shape.radius, 20, color=obj.color2, filled=False, linewidth=2).add_attr(t)
                else:
                    path = [trans*v for v in f.shape.vertices]
                    self.viewer.draw_polygon(path, color=obj.color1)
                    path.append(path[0])
                    self.viewer.draw_polyline(path, color=obj.color2, linewidth=2)

        for x in [self.helipad_x1, self.helipad_x2]:
            flagy1 = self.helipad_y
            flagy2 = flagy1 + 50/SCALE
            self.viewer.draw_polyline([(x, flagy1), (x, flagy2)], color=(1, 1, 1))
            self.viewer.draw_polygon([(x, flagy2), (x, flagy2-10/SCALE), (x + 25/SCALE, flagy2 - 5/SCALE)],
                                     color=(0.8, 0.8, 0))

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


class LunarLanderMultiContinuous(LunarLanderMulti):
    continuous = True
