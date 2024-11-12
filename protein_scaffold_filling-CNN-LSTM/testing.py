from graphviz import Digraph

dot = Digraph(comment='The GPT Model')

# Add nodes
dot.node('I', 'Input')
dot.node('E', 'Embedding')
dot.node('T1', 'Transformer Block 1')
dot.node('T2', 'Transformer Block 2')
dot.node('TD', '...')
dot.node('TN', 'Transformer Block N')
dot.node('O', 'Output Layer')
dot.node('P', 'Predicted Sequence')

# Add edges
dot.edges([('I', 'E'), ('E', 'T1')])
dot.edge('T1', 'T2', constraint='false')
dot.edge('T2', 'TD', constraint='false')
dot.edge('TD', 'TN', constraint='false')
dot.edges([('TN', 'O'), ('O', 'P')])

# Render the graph to a file (e.g., a PNG image)
dot.render('gpt_model_diagram', format='png', cleanup=True)
