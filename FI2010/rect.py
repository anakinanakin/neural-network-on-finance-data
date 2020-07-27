import matplotlib.pyplot as plt
from matplotlib import patches

def trans2rect(arr):
	tarr = []
	trend = arr[0]
	width = 1
	day = 0
	for elm in arr[1:]:
		if elm == trend:
			width += 1
		else:
			tarr.append((trend, day, width))
			trend = elm
			day  += width
			width = 1
	tarr.append((trend, day, width))
	return tarr

# ax = plt.subplot(111)
# ax.add_patch(patches.Rectangle((0,0), 1,3, color=(.8, .2, .2)))
# ax.add_patch(patches.Rectangle((1,2), 2,2, color=(.2, .8, .2)))
ax = plt.subplot(111)
# plt.ylim(0, 5)
plt.xlim(0, 10)

pred = [1,1,0,0,1,0,0,1,0,0]


ans = [1,1,1,1,0,0,0,1,1,1]
tans = [(1,0,4),(0,4,3),(1,7,3)]
tpred = trans2rect(pred)

for a in tans:
	col = (.8, .2, .2) if a[0] == 1 else (.2, .8, .2)
	ax.add_patch(patches.Rectangle((a[1],0), a[2],0.5, color=col))

for a in tpred:
	col = (.8, .2, .2) if a[0] == 1 else (.2, .8, .2)
	ax.add_patch(patches.Rectangle((a[1],0.5), a[2],0.5, color=col))


#plt.savefig('2.png')
plt.show()
