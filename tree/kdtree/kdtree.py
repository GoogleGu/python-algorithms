class TreeNode:

    def __init__(self, val, d):
        self.val = val
        self.d = d
        self.left = None
        self.right = None


class KDTree:

    def __init__(self, points):
        self.root = None
        self.k = len(points[0])
        for point in points:
            self.insert(point)

    def insert(self, point):
        def _insert(node, d):
            d = d % self.k
            if node is None:
                node = TreeNode(point, d)
            elif node.val == point:
                raise ValueError("Found duplicate point!")
            elif point[d] < node.val[d]:
                _insert(node.left, d+1)
            else:
                _insert(node.right, d+1)

        _insert(self.root, 0)

    def search(self, point):
        def _search(node, d):
            d = d % self.k
            if node is None:
                node = TreeNode(point, d)
            elif node.val == point:
                return node.val
            elif point[d] < node.val[d]:
                return _search(node.left, d+1)
            else:
                return _search(node.right, d+1)

        return _search(self.root, 0)

    def delete(self, point):
        # TODO
        deleted_node = self.search(point)
        if deleted_node is None:
            raise ValueError('point {} is not in this kd-tree'.format(point))
        if deleted_node.right is not None:
            right_min = self._find_min(deleted_node.right, deleted_node.d, deleted_node.d+1)

    def find_min(self, target_dim):
        return self._find_min(self.root, target_dim, 0)

    def _find_min(self, node, target_dim, d):
        d = d % self.k
        if node is None:
            return None
        if target_dim == d:
            return self._find_min(node.left, target_dim, d + 1) or node.val
        else:
            left_min = self._find_min(node.left, target_dim, d + 1) or node.val
            right_min = self._find_min(node.right, target_dim, d + 1) or node.val
            return self.minimum(left_min, right_min, dim=target_dim)

    def k_nearest_of(self, point):
        pass

    @staticmethod
    def minimum(point1, point2, dim):
        if point1 is None:
            return point2
        if point2 is None:
            return point1
        if point1[dim] > point2[dim]:
            return point2
        else:
            return point1