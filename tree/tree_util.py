def print_binary_tree(root):

    if root is None:
        return []

    nodes = [[root]]

    while True:
        last_row = nodes[-1]
        this_row = []
        for node in last_row:
            if node.left:
                this_row.append(node.left)
            if node.right:
                this_row.append(node.right)
        if len(this_row) == 0:
            break
        else:
            nodes.append(this_row)
    
    result = []
    for row in nodes:
        result.append([node.val for node in row])
    
    return result