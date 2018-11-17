import numpy as np
from heapq import heappush, heappop
from animation import draw

class Node():
    """
    cost_from_start - the cost of reaching this node from the starting node
    state - the state (row,col)
    parent - the parent node of this node, default as None
    """
    def __init__(self, state, cost_from_start, parent = None):
        self.state = state
        self.parent = parent
        self.cost_from_start = cost_from_start


class EightPuzzle():
    
    def __init__(self, start_state, goal_state, algorithm, array_index):
        self.start_state = start_state
        self.goal_state = goal_state
        self.visited = [] 
        self.algorithm = algorithm
        self.array_index = array_index

    # goal test function
    def goal_test(self, current_state):
        # your code goes here:
        return np.array_equal(current_state, self.goal_state)

    # get cost function
    def get_cost(self, current_state, next_state):
        # your code goes here:
        return 1

    # get successor function
    def get_successors(self, state):
        successors = []
        # your code goes here:
        moves = [[1, -1, 0, 0],[0, 0, -1, 1]]

        x,y = np.where(state == 0)
        
        for i in range(0, 4):
            candidate_x = x[0] + moves[0][i]
            candidate_y = y[0] + moves[1][i]
            if (candidate_x >= 0 and candidate_x < 3 and candidate_y >= 0 and candidate_y < 3):
                temp = state.copy()
                temp[candidate_x][candidate_y] = state[x[0],y[0]]
                temp[x[0]][y[0]] = state[candidate_x][candidate_y]
                successors.append(temp)

        return successors

    # get priority of node for UCS
    def priority(self, node):
        priority = node.cost_from_start

        for i in range(0,3):
            for j in range(0,3):
                element = self.goal_state[i, j]
                row, col = np.where(node.state == element)

                priority += abs(i - row)
                priority += abs(j - col)

        return priority

    # draw 
    # you do not need to modify anything in this function.
    def draw(self, node):
        path=[]
        while node.parent:
            path.append(node.state)
            node = node.parent
        path.append(self.start_state)
        draw(path[::-1], self.array_index, self.algorithm)

    # solve it
    def solve(self):
        container = [] # node
        count = 1
        state = self.start_state.copy()
        node = Node(state, 0, None)
        self.visited.append(state)

        self.get_successors(node.state)
        if self.algorithm == 'Depth-Limited-DFS': 
            # your code goes here:
            container.append(node)

        elif self.algorithm == 'BFS': 
            # your code goes here:
            container.append(node)

        elif self.algorithm == 'UCS': 
            # your code goes here:
            heappush(container, (count, count, node))

        while container:
            # your code goes here:

            if self.algorithm == 'UCS': 
                node = heappop(container)[2]

            else:
                node = container.pop()
            
            if self.goal_test(node.state):
                self.draw(node)
                break

            successors = self.get_successors(node.state)
            for next_state in successors:

                visited = False
                for s in self.visited:
                    if np.array_equal(s, next_state):
                        visited = True
                
                if not visited:
                    next_cost = node.cost_from_start + self.get_cost(node.state, next_state)
                    next_node = Node(next_state, next_cost, node)
                    self.visited.append(next_state)

                    count += 1
                    if self.algorithm == 'Depth-Limited-DFS' and next_cost < 16:
                        container.append(next_node)
                    elif self.algorithm == 'UCS':
                        heappush(container, (self.priority(next_node), count, next_node))
                    else:
                        container.append(next_node)
                    

# You do not need to change anything below this line, except for debuggin reason.
if __name__ == "__main__":
    
    goal = np.array([[1,2,3],[4,5,6],[7,8,0]])

    start_arrays = [np.array([[0,1,3],[4,2,5],[7,8,6]]), 
                    np.array([[0,2,3],[1,4,6],[7,5,8]])] 

    algorithms = ['Depth-Limited-DFS', 'BFS', 'UCS']
    
    for i in range(len(start_arrays)):
        for j in range(len(algorithms)):
            game = EightPuzzle(start_arrays[i], goal, algorithms[j], i )
            game.solve()
