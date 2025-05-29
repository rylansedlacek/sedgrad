from graphviz import Digraph
import matplotlib.pyplot as plt

def trace(root): # traces back all nodes and edges from the root
    nodes = set()
    edges = set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    
    build(root)
    return nodes, edges


def draw_dot(root): # creates the visual represtation of the engine computation graph
     dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})
     nodes, edges = trace(root)

     for n in nodes:
        uid = str(id(n))
        label = f"{n._op}\n{n.data:.4f}\ngrad={n.grad:.4f}"
        dot.node(name=uid, label=label, shape='record')

        if n._op:
            dot.node(name=uid + n._op, label=n._op, shape='circle')
            dot.edge(uid + n._op, uid)

     for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

     return dot



def visualize_network(mlp, input_size):
    layer_sizes = [input_size] + [len(layer.neurons) for layer in mlp.layers]

    fig, ax = plt.subplots()
    v_spacing = 1.0 / float(max(layer_sizes))
    h_spacing = 1.0 / float(len(layer_sizes) - 1)

    # Draw neurons
    for i, layer_size in enumerate(layer_sizes):
        for j in range(layer_size):
            x = i * h_spacing
            y = 1 - j * v_spacing
            circle = plt.Circle((x, y), v_spacing / 4, color='black', fill=False)
            ax.add_artist(circle)

    # Draw connections (arrows)
    for i in range(len(layer_sizes) - 1):
        for j in range(layer_sizes[i]):
            for k in range(layer_sizes[i + 1]):
                x0 = i * h_spacing
                y0 = 1 - j * v_spacing
                x1 = (i + 1) * h_spacing
                y1 = 1 - k * v_spacing
                ax.plot([x0, x1], [y0, y1], 'k-', lw=0.5)

    ax.set_aspect('equal')
    ax.axis('off')
    plt.show()


def visualize_network(mlp, input_size):
    layer_sizes = [input_size] + [len(layer.neurons) for layer in mlp.layers]

    fig, ax = plt.subplots()
   

    v_spacing = 1.0 / float(max(layer_sizes))
    h_spacing = 1.0 / float(len(layer_sizes) - 1)

    # draw the neurons here, layer by layer which we get specificed
    for i, layer_size in enumerate(layer_sizes):
        for j in range(layer_size):
            x = i * h_spacing
            y = 1 - j * v_spacing
            circle = plt.Circle((x, y), v_spacing / 4, color='black', fill=False)
            ax.add_artist(circle)

    # draw the connections to said neurons
    for i in range(len(layer_sizes) - 1):
        for j in range(layer_sizes[i]):
            for k in range(layer_sizes[i + 1]):
                x0 = i * h_spacing
                y0 = 1 - j * v_spacing
                x1 = (i + 1) * h_spacing
                y1 = 1 - k * v_spacing
                ax.plot([x0, x1], [y0, y1], 'k-', lw=0.5)

    ax.set_aspect('equal')
    ax.axis('off')
    plt.show()
