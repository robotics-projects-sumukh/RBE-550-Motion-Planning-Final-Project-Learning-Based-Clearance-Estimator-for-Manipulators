"""Planning for MotionBenchMaker problems.
"""
from fire import Fire
from abc import ABC
from grapeshot.assets import ROBOTS
from grapeshot.assets.environments import PLANE
from grapeshot.extensions.octomap import load_mbm_octomap
from grapeshot.model.world import World
from grapeshot.model.robot import process_srdf
from grapeshot.planning.context import get_OMPL_context
from grapeshot.planning.goals import getJointBoundGoal
from grapeshot.planning.moveit import process_moveit_request
from grapeshot.planning.trajectory import path_to_trajectory
from grapeshot.simulators.pybullet import PyBulletSimulator
from grapeshot.planning.context import get_group_state_space, get_OMPL_statespace
from grapeshot.model.skeleton import JointType
from grapeshot.model.world import EnvironmentBuilder
from grapeshot.model.environment import process_environment_yaml

from grapeshot.planning.validity import ValidityChecker, CollisionTrackingValidityChecker, ContactIgnoringValidityChecker # Make sure to import the validity checker
from grapeshot.planning.objectives import PathLengthObjective, ClearanceObjective

import numpy as np
import ompl.base._base as ob
import ompl.geometric._geometric as og
from grapeshot.planning.typing import SetStateFn
from grapeshot.model.group import Group, GroupABC, GroupSet
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import StandardScaler
from grapeshot.planning.typing import SetStateFn
import xml.etree.ElementTree as ET

INVERSE_TRANSFORM_FACTOR = 0.5941515957852193

# Define the neural network model
class DistancePredictor(nn.Module):   
    def __init__(self, input_size=14, hidden_layers=[1400, 1400], output_size=1): 
        super(DistancePredictor, self).__init__()
        
        # Define the network layers
        layers = []
        in_features = input_size
        for hidden_units in hidden_layers:
            layers.append(nn.Linear(in_features, hidden_units))
            layers.append(nn.ReLU())  # Activation function
            # add a dropout layer
            layers.append(nn.Dropout(0.01))
            in_features = hidden_units
        layers.append(nn.Linear(in_features, output_size))  # Output layer

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

print("Model Loading")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DistancePredictor().to(device)
model.load_state_dict(torch.load('model/model_cnrrt_1_obs.pth', map_location=device))
print("Model loaded")


def predict_distance(model, s: ob.State, position: list[list[float]], sphere: bool) -> float:
    j11 = s[0][0] / (0.38615) 
    j12 = s[0][1] / (1.6056)
    j13 = s[0][2] / (1.518)
    j14 = s[0][3] / (3.14)
    j15 = s[0][4] / (2.251)
    j2 = s[1].value / (3.14)
    j3 = s[2].value / (2.16)
    j4 = s[3].value / (3.14)

    if sphere:
        obstacle_x = position[0][0]
        obstacle_y = position[0][1]
        obstacle_z = position[0][2]
    else:
        obstacle_x = position[1][0]
        obstacle_y = position[1][1]
        obstacle_z = position[1][2]

    obstacle_length = 0.3
    obstacle_breadth = 0.3
    obstacle_height = 0.3

    # Prepare input data and transfer it to GPU
    input_data = [j11, j12, j13, j14, j15, j2, j3, j4, obstacle_x, obstacle_y, obstacle_z, obstacle_length, obstacle_breadth, obstacle_height]
    input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Predict using the model
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    return output.item() * INVERSE_TRANSFORM_FACTOR



class CustomValidityChecker(ob.StateValidityChecker):
    world: World           # Simulator for collision checking
    set_state: SetStateFn  # Function for setting world state from OMPL state

    def __init__(self, si: ob.SpaceInformation, world: World, set_state: SetStateFn, positions: list[list[float]]):
        super().__init__(si)
        self.world = world
        self.set_state = set_state
        self.positions = positions

    def isValid(self, state: ob.State) -> bool:
        self.set_state(state, self.world)
        return not self.world.in_collision() and predict_distance(model, state, self.positions, True) > 0.05


class CustomPathLengthObjective(ob.PathLengthOptimizationObjective):
    groups: list[GroupABC]
    def __init__(self, si: ob.SpaceInformation, world: World, set_state: SetStateFn, groups: list[GroupABC],  positions: list[list[float]]):
        super().__init__(si)
        self.group = groups[0]
        self.world = world
        self.set_state = set_state
        self.svc = si.getStateValidityChecker()  # State Validity Checker to check collisions and clearance
        self.si = si
        self.positions = positions

    def motionCost(self, s1: ob.State, s2: ob.State) -> ob.Cost:
        self.set_state(s1, self.world)
        clearance_state1_1 = predict_distance(model, s1, self.positions, True)
        clearance_state1_2 = predict_distance(model, s1, self.positions, False)
        clearance_state1 = min(clearance_state1_1, clearance_state1_2)
        cost1 = 1 / (clearance_state1 + 1e-4)
        self.set_state(s2, self.world)
        clearance_state2_1 = predict_distance(model, s2, self.positions, True)
        clearance_state2_2 = predict_distance(model, s2, self.positions, False)
        clearance_state2 = min(clearance_state2_1, clearance_state2_2)
        cost2 = 1 / (clearance_state2 + 1e-4)
        avg = (cost1 + cost2) / 2
        cost = avg * self.si.distance(s1, s2)
        return ob.Cost(cost)
    
    def stateCost(self, state: ob.State) -> ob.Cost:
        # Set the robot's state in the world for clearance checking
        self.set_state(state, self.world)
        # Compute the clearance from obstacles in the environment (higher clearance = safer path)
        clearance_1 = predict_distance(model, state, self.positions, True)
        clearance_2 = predict_distance(model, state, self.positions, False)
        clearance = min(clearance_1, clearance_2)
        # A small epsilon added to avoid division by zero and to ensure cost is finite
        clearance_cost = 1 / (clearance + 1e-4)  # Inverse of clearance for safety penalty
        # Return the clearance-based cost. You can still combine this with other factors (e.g., distance) if needed.
        return ob.Cost(clearance_cost)

def get_motion_path_cost(path: og.PathGeometric, si: ob.SpaceInformation, sc: ValidityChecker, positions: list[list[float]]):
    cost = 0
    cost_og = 0
    for i in range(path.getStateCount()-1):
        s1 = path.getState(i)
        s2 = path.getState(i+1)
        clearance_state_og =  sc.clearance(s1)
        clearance_state_1 = predict_distance(model, s1, positions, True)
        clearance_state_2 = predict_distance(model, s1, positions, False)
        clearance_state = min(clearance_state_1, clearance_state_2)

        if clearance_state_og < 0.05:
            continue
        if clearance_state < 0.05:
            continue
        clearance_state1_og = sc.clearance(s1)
        cost1_og = 1 / (clearance_state1_og + 1e-4)
        clearance_state2_og = sc.clearance(s2)
        cost2_og = 1 / (clearance_state2_og + 1e-4)
        avg_og = (cost1_og + cost2_og) / 2
        cost_og += avg_og * si.distance(s1, s2)

        clearance_state1_1 = predict_distance(model, s1, positions, True)
        clearance_state1_2 = predict_distance(model, s1, positions, False)
        clearance_state1 = min(clearance_state1_1, clearance_state1_2)
        cost1 = 1 / (clearance_state1 + 1e-4)
        clearance_state2_1 = predict_distance(model, s2, positions, True)
        clearance_state2_2 = predict_distance(model, s2, positions, False)
        clearance_state2 = min(clearance_state2_1, clearance_state2_2)
        cost2 = 1 / (clearance_state2 + 1e-4)
        avg = (cost1 + cost2) / 2
        cost += avg * si.distance(s1, s2)
    
    return [cost, cost_og]


def main(
        robot: str = "fetch",          # Robot to load. See assets/ folder for all robots.
        problem: str = "table_pick",   # Problem to solve. See assets/ folder for problems availble for robots.
        planner: str = "RRTstar",       # Planner to use. Anything in the ompl.geometric namespace.
        load_plane: bool = True,      # Load the ground plane.
        visualize: bool = True,       # Visualize with GUI.
        simplify: bool = False,        # Simplify the path.
        resolution: float = 0.03,     # Validity checking resolution for planning.
        speedup: float = 3,            # Speedup to animation when displaying
        benchmark: bool = False,       # Rather than planning, benchmark this planning problem.
        runs: int = 10,                # Number of runs to use in benchmarking.
        sensed: bool = False,          # Use the sensed representation of the problem if available.
        show_goal: bool = True,        # Show goal state
        validity_checker_type: str = "basic"
    ):
                                       # Get the requested robot and problem
    robot_resource = ROBOTS[robot]
    problem_resource = robot_resource.problems[problem]
    # Set the position of the sphere and cube
    position_sphere = [0.0,0.0,0.0]
    position_cube = [0.0,0.0,0.0]
    positions = [position_sphere, position_cube]
    # Create simulation with robot, load and process URDF and SRDF.
    # NOTE: Need to sometimes disable the contact threshold in PyBullet due to the proximity of the problem's
    #       queries to obstacle geometry. Specifically, the Fetch tuck pose.
    world = World(PyBulletSimulator(visualize))
    skel = world.add_skeleton(robot_resource.urdf)
    groups = process_srdf(skel, robot_resource.srdf)

    # Load sensed environment. Needs OctoMap extension installed.
    if sensed and problem_resource.sensed:
        octomap = load_mbm_octomap(problem_resource.sensed)
        world.add_environment_builder(octomap.get_builder())

    # Load modeled environment
    else:
        env = process_environment_yaml(problem_resource.environment)
        world.add_environment_builder(env)
        env_string = str(env)

        # Parse the XML string
        root = ET.fromstring(env_string)

        # Extract xyz values from all <origin> tags
        origin_xyz_values = []
        for origin in root.findall(".//origin"):
            xyz = origin.get("xyz")
            if xyz:
                # Convert the xyz string to a list of floats
                xyz_values = list(map(float, xyz.split()))
                origin_xyz_values.append(xyz_values)

        # Output the result
        position_sphere = [origin_xyz_values[0][0], origin_xyz_values[0][1], origin_xyz_values[0][2]]
        position_cube = [origin_xyz_values[2][0], origin_xyz_values[2][1], origin_xyz_values[2][2]]
        positions = [position_sphere, position_cube]

    # Load ground plane.
    if load_plane:
        plane = world.add_skeleton(PLANE)
        world.acm.disable_skeleton(plane)  # Disable collision checking with the plane

    # Setup collision checking with environment.
    world.setup_collision_filter()

    # Load motion planning request from MotionBenchMaker's format (MoveIt MotionPlanningRequest message in YAML)
    request = process_moveit_request(skel, groups, problem_resource.request)

    # Set state to initial from request.
    world.set_joint_positions(request.initial_state)

    # Get planning context for specified planning group.
    context = get_OMPL_context(world, [request.group], planner)
    context.planner.setRange(1.4)
    context.planner.setGoalBias(0.1)

    # Get starting scoped state from current world state.
    start = context.scoped_state_from_world(world)


    #make a new goal state
    # goal_state = context.scoped_state()
    # goal_state[0] = np.random.uniform(-0.38615, 0.38615)
    # goal_state[1] = np.random.uniform(-1.6056, 1.6056)
    # goal_state[2] = np.random.uniform(-1.221, 1.518)
    # goal_state[3] = np.random.uniform(-3.1416, 3.1416)
    # goal_state[4] = np.random.uniform(-2.251, 2.251)
    # goal_state[5] = np.random.uniform(-3.1416, 3.1416)
    # goal_state[6] = np.random.uniform(-2.16, 2.16)
    # goal_state[7] = np.random.uniform(-3.1416, 3.1416)   


    goal_state = context.scoped_state()
    goal = getJointBoundGoal(context, request)
    goal.sampleGoal(goal_state())
    

    if show_goal:
        context.set_state(goal_state(), world)

    else:
        context.set_state(start(), world)

    # Instantiate the specified validity checker
    if validity_checker_type == "basic":
        validity_checker = CustomValidityChecker(context.setup.getSpaceInformation(), world, context.set_state, positions)
        validity_checker_original = ValidityChecker(context.setup.getSpaceInformation(), world, context.set_state)
    else:
        raise ValueError("Unknown validity checker type")
    context.set_validity_checker(validity_checker)

    # Set the optimization objective
    custom_path_length_objective = CustomPathLengthObjective(context.si, world, context.set_state, [request.group], positions)
    context.setup.setOptimizationObjective(custom_path_length_objective)
    print("Optimization objective set in the context.")

    # Benchmark the request, if asked.
    if benchmark:
        context.setup.setStartAndGoalStates(start, goal_state)

        # Add additional parameters either to entire experiment
        # or as a function of the planner and path per run of the planner
        additional_properties = {
            'resolution': resolution,
            'min_clearance': lambda _,
            path: min(context.validity_checker.clearance(s) for s in path.getStates())
            }

        # Run benchmarking.
        context.benchmark(
            f'{robot}_{problem}',
            request.planning_time,
            runs = runs,
            simplify = simplify,
            additional_properties = additional_properties
            )

    else:
        plan_num = 0
        while plan_num < 10:
            # Clear existing planning data structures from prior query.
            # If the planner supports multiple queries, this will keep around relevant information, e.g., PRM
            # context.clear_query()
            context.setup.setStartAndGoalStates(start, goal_state)
            # Do motion planning.
            if context.plan(request.planning_time):
                print(f"Plan {plan_num} found.")
                plan_num += 1
                path = context.get_solution_path(simplify = simplify)
                # print("Original Path Cost: ", get_path_cost(path, context.si, validity_checker_original))
                cost, cost_og = get_motion_path_cost(path, context.si, validity_checker_original, positions)
                print("Original Path Cost: ", cost_og)
                print("Model Path Cost: ", cost)
                
                print(f"Path length: {path.length()}")


                # If visualizing, animate the trajectory in the GUI.
                if visualize:
                    # Get the solution trajectory.
                    path = context.get_solution_path(simplify = simplify)
                    # Convert the waypoints into an executable trajectory.
                    traj = path_to_trajectory(context, path)
                    # Animate.
                    world.animate(traj, speedup = speedup)

                # Swap start and goal states and continue planning.
                temp = context.scoped_state()
                context.ss.copyState(temp(), start())
                context.ss.copyState(start(), goal_state())
                context.ss.copyState(goal_state(), temp())
            else:
                print("Failed to find exact solution. Trying again.")
                continue

if __name__ == '__main__':
    Fire(main)
