

class TreeNode:

    def __init__(self, val, height=0):
        self.val = val
        self.left = None
        self.right = None
        self.parent = None
        self.height = height
    
    def find(self, val):
        if self.val == val:
            return self
        elif self.val > val:
            if self.left is None:
                return None
            return self.find(self.left)
        else:
            if self.right is None:
                return None
            return self.find(self.right)
    
    def insert(self, node):

        if self.val > node.val:
            if self.left is None:
                self.left = node
                node.parent = self
            else:
                self.left.insert(node)
        else:
            if self.right is None:
                self.right = node
                node.parent = self
            else:
                self.right.insert(node)
        update_height(self)

def height(node):
    if node is None:
        return -1    
    else:
        return node.height

def update_height(node):
    node.height = max([height(node.left), height(node.right)]) + 1

class AVLTree:
    """
    只实现了插入的自平衡，没有实现删除方法和相应的自平衡。
    """

    def __init__(self):
        self.root = None
        self.node_nums = 0
    
    def find(self, val):
        if not self.root:
            return None
        else:
            return self.root.find(val)

    def insert(self, val):
        self.node_nums += 1
        new_node = TreeNode(val)
        if self.root is None:
            self.root = new_node
        else:
            self.root.insert(new_node)
        self.rebalance(new_node)

    def rebalance(self, node):
        while node is not None:
            update_height(node)
            if height(node.left) - height(node.right) > 1:
                self.right_rotate(node)
            elif height(node.left) - height(node.right) < -1:
                self.left_rotate(node)
            node = node.parent

    def right_rotate(self, x_node):
        root = x_node.parent
        l_node = x_node.left
        lr_node = l_node.right

        x_node.left = lr_node
        if lr_node is not None:
            lr_node.parent = x_node

        l_node.right = x_node
        x_node.parent = l_node

        l_node.parent = root
        if root is None:
            self.root = l_node
        else:
            if self.root.left is x_node:
                self.root.left = l_node
            else:
                self.root.right = l_node
        update_height(x_node)
        update_height(l_node)

    def left_rotate(self, x_node):
        root = x_node.parent
        r_node = x_node.right
        rl_node = r_node.left

        x_node.right = rl_node
        if rl_node is not None:
            rl_node.parent = x_node

        r_node.left = x_node
        x_node.parent = r_node

        r_node.parent = root
        if root is None:
            self.root = r_node
        else:
            if self.root.left is x_node:
                self.root.left = r_node
            else:
                self.root.right = r_node
        
        update_height(x_node)
        update_height(r_node)


if __name__ == "__main__":
    avl = AVLTree()
    avl.insert(5)
    avl.insert(4)
    avl.insert(3)
    avl.insert(2)
    avl.insert(1)
    avl.insert(0)
