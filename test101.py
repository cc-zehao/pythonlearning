import networkx as nx
import matplotlib.pyplot as plt
G=nx.dodecahedral_graph()
nx.draw(G)
plt.draw()
# pos = nx.spring_layout(G)
#
# nx.draw_networkx(G, pos=pos, nodelist=["A", "B"], node_size=1000, node_color='#A0CBE2', font_size=20, width=2)
#
# nx.draw_networkx(G, pos=pos, nodelist=["C"], node_size=2000, node_color='#FF0000', font_size=20, width=2,)
#
# nx.draw_networkx(G, pos=pos, nodelist=["D"], node_size=3000, node_color='#FFFF00', font_size=20, width=2)
