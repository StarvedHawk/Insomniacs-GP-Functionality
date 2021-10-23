from matplotlib import pyplot as plt

fig = plt.figure()
fig.set_dpi(100)
fig.set_size_inches(7, 6.5)

ax = plt.axes(xlim=(0, 1), ylim=(0, 1))
patch = plt.Polygon([[0.1,0.1],[0.3,0.1],[0.3,0.3],[0.1,0.3]])
ax.add_patch(patch)
plt.show()