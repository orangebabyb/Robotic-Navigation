
import cv2
import numpy as np

from path_planning import *
from path_planning.a_star_planner import AStarPlanner


class AStarImplementation(AStarPlanner):
    # TODO: implement your own version of preloop, step and postloop
    def preloop(self):
        import heapq
        from itertools import count
        self.counter = count()
        self.queue = []
        self.g = {}
        self.visited_nodes = set()
        
        self.g[self.start_node] = 0
        f_score = calculate_node_distance(self.start_node, self.goal_node)
        heapq.heappush(self.queue, (f_score, next(self.counter), self.start_node))

    def step(self):
        import heapq
        if not self.queue:
            self.is_done.set()
            return
            
        current_node = heapq.heappop(self.queue)[2]
        if current_node in self.visited_nodes:
            return
            
        self.visited_nodes.add(current_node)
        
        if calculate_node_distance(current_node, self.goal_node) < self.goal_threshold:
            self.goal_node.parent = current_node
            self.is_done.set()
            return
            
        for neighbor in self.get_neighbor_nodes(current_node):
            if neighbor in self.visited_nodes:
                continue
                
            tentative_g = self.g.get(current_node, float('inf')) + calculate_node_distance(current_node, neighbor)
            
            if tentative_g < self.g.get(neighbor, float('inf')):
                neighbor.parent = current_node
                self.g[neighbor] = tentative_g
                f_score = tentative_g + calculate_node_distance(neighbor, self.goal_node)
                heapq.heappush(self.queue, (f_score, next(self.counter), neighbor))

    def postloop(self):
        return (
            collect_path(self.goal_node), 
            self.visited_nodes
        )