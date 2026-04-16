
import cv2
import numpy as np

from path_planning import *
from path_planning.rrt_star_planner import RRTStarPlanner


class RRTStarImplementation(RRTStarPlanner):
    # TODO: implement your own version of preloop, step and postloop
    def preloop(self):
        self.visited_nodes = set()
        self.start_node.cost = 0
        self.visited_nodes.add(self.start_node)
        
        self.children = {self.start_node: set()}
        self.best_goal_cost = float('inf')

    def propagate_cost(self, node):
        for child in self.children.get(node, []):
            dist = calculate_node_distance(node, child)
            child.cost = node.cost + dist
            self.propagate_cost(child)

    def step(self):
        random_node = self.sample_random_node()
        if not self.visited_nodes:
            return
            
        nearest_node = min(self.visited_nodes, key=lambda n: calculate_node_distance(n, random_node))
        
        dist = calculate_node_distance(nearest_node, random_node)
        if dist > self.step_size:
            ratio = self.step_size / dist
            diff_x = random_node.coordinates.x - nearest_node.coordinates.x
            diff_y = random_node.coordinates.y - nearest_node.coordinates.y
            new_x = nearest_node.coordinates.x + diff_x * ratio
            new_y = nearest_node.coordinates.y + diff_y * ratio
            new_node = PathNode(coordinates=PixelCoordinates(new_x, new_y))
        else:
            new_node = PathNode(coordinates=random_node.coordinates)
            
        if not check_inside_map(self.occupancy_map, new_node):
            return
        if not check_collision_free(self.occupancy_map, nearest_node, new_node):
            return
            
        neighbors = []
        for n in self.visited_nodes:
            if calculate_node_distance(n, new_node) <= self.search_radius:
                if check_collision_free(self.occupancy_map, n, new_node):
                    neighbors.append(n)
                    
        new_node.parent = nearest_node
        new_node.cost = nearest_node.cost + calculate_node_distance(nearest_node, new_node)
        
        for neighbor in neighbors:
            cost = neighbor.cost + calculate_node_distance(neighbor, new_node)
            if cost < new_node.cost:
                new_node.parent = neighbor
                new_node.cost = cost
                
        self.visited_nodes.add(new_node)
        self.children[new_node] = set()
        if new_node.parent is not None:
            self.children[new_node.parent].add(new_node)
        
        for neighbor in neighbors:
            cost = new_node.cost + calculate_node_distance(new_node, neighbor)
            if cost < neighbor.cost:
                old_parent = neighbor.parent
                if old_parent is not None and old_parent in self.children and neighbor in self.children[old_parent]:
                    self.children[old_parent].remove(neighbor)
                neighbor.parent = new_node
                self.children[new_node].add(neighbor)
                neighbor.cost = cost
                self.propagate_cost(neighbor)
                
        goal_dist = calculate_node_distance(new_node, self.goal_node)
        if goal_dist < self.goal_threshold:
            if check_collision_free(self.occupancy_map, new_node, self.goal_node):
                cost_to_goal = new_node.cost + goal_dist
                if cost_to_goal < self.best_goal_cost:
                    self.goal_node.parent = new_node
                    self.best_goal_cost = cost_to_goal
                    self.is_done.set()

    def postloop(self):
        return (
            collect_path(self.goal_node), 
            self.visited_nodes
        )