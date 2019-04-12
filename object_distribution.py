'''
determine all the unique objects in the SUNRGBD dataset
and plot a histogram of occurances of each object
'''
import utils
import pickle
import operator
import matplotlib.pyplot as plt
from matplotlib import rcParams
import json

meta_data = utils.load_SUNRGBD_meta()
# get unique objects
items = 7751
uniqu = set()
for entry in range(items):
    labels = utils.get_label(meta_data, entry)
    for objekt in range(len(labels)):
        uniqu.add(labels[objekt])

# set up dict as a counter
uniqu_list= list(uniqu)
counter = {}
for item in uniqu_list:
    counter[item] = 0

# count the items and add to dicts
for entry in range(items):
    labels = utils.get_label(meta_data, entry)
    for objekt in range(len(labels)):
        counter[labels[objekt]] += 1

# order the dictionary
ordered_count = sorted(counter.items(),key=operator.itemgetter(1),reverse=True)
ordered_count = dict(ordered_count)
# print out item counts
print(json.dumps(dict(sorted(counter.items(),key=operator.itemgetter(1))), indent=4))
print('Num of labels :'+str(len(uniqu)))
# plot a bar chart
fig, ax = plt.subplots()
# set up latex font
# rcParams['mathtext.fontset'] = 'stix'
# rcParams['font.family'] = 'STIXGeneral'
# plot
ax.bar(range(len(ordered_count)), list(ordered_count.values()), align='center')
# ax.xticks(range(len(ordered_count)), list(ordered_count.keys()))   # too many to do this
ax.set_yscale('log')
ax.set_xlabel('Unique Objects')
ax.set_ylabel('Number of Objects')
plt.show()

#    now do for test data
# get unique objects
uniqu = set()
for entry in range(items, 10335):
    labels = utils.get_label(meta_data, entry)
    for objekt in range(len(labels)):
        uniqu.add(labels[objekt])

# set up dict as a counter
uniqu_list= list(uniqu)
counter = {}
for item in uniqu_list:
    counter[item] = 0

# count the items and add to dicts
for entry in range(items, 10335):
    labels = utils.get_label(meta_data, entry)
    for objekt in range(len(labels)):
        counter[labels[objekt]] += 1

# order the dictionary
ordered_count = sorted(counter.items(),key=operator.itemgetter(1),reverse=True)
ordered_count = dict(ordered_count)
# print out item counts
print(json.dumps(dict(sorted(counter.items(),key=operator.itemgetter(1))), indent=4))
print('Num of labels :'+str(len(uniqu)))
# plot a bar chart
fig, ax = plt.subplots()
# set up latex font
# rcParams['mathtext.fontset'] = 'stix'
# rcParams['font.family'] = 'STIXGeneral'
# plot
ax.bar(range(len(ordered_count)), list(ordered_count.values()), align='center')
# ax.xticks(range(len(ordered_count)), list(ordered_count.keys()))   # too many to do this
ax.set_yscale('log')
ax.set_xlabel('Unique Objects')
ax.set_ylabel('Number of Objects')
plt.show()
