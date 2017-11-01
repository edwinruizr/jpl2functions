# import matplotlib.pyplot as plt
# import numpy as np

# import plotly.plotly as py
# # Learn about API authentication here: https://plot.ly/python/getting-started
# # Find your api_key here: https://plot.ly/settings/api

# gaussian_numbers = np.random.randn(1000)
# plt.hist(gaussian_numbers)
# plt.title("Gaussian Histogram")
# plt.xlabel("Value")
# plt.ylabel("Frequency")

# fig = plt.gcf()

# plot_url = py.plot_mpl(fig, filename='mpl-basic-histogram')




import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.axes as ax

mu, sigma = 20, 5
x = mu + sigma*np.random.randn(10000)
#actual data
#x = 200, 2005

# the histogram of the data
#n, bins, patches = plt.hist(x, 50, normed=1, facecolor='green', alpha=0.75)
#face color = what color your graph will be

n, bins, patches = plt.hist(x, 10, normed=1, facecolor='blue')
#defines limit for the x axis
plt.xlim(xmin = 0, xmax = 20)

#defines limit for y axis
plt.ylim(ymin = 0, ymax = 1)

# add a 'best fit' line
#y = mlab.normpdf( bins, mu, sigma)

#just plots best fit line
#l = plt.plot(bins, y, 'r--', linewidth=1)

#x label
plt.xlabel('Smarts')
#y label
plt.ylabel('Probability')
#title
plt.title('Example')
#plt.axis([40, 160, 0, 0.03])

#adds a grid
plt.grid(True)

plt.show()
