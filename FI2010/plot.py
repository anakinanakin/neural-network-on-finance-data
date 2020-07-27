import sys
import os 
import pandas as pd
import numpy as np
import psycopg2, psycopg2.extras
import datetime
from datetime import timedelta
import matplotlib
matplotlib.use('Agg')
from matplotlib import patches
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure




ax1 = plt.subplot2grid((10,4),(0,0), colspan=2, rowspan=2)
ax1.text(0,0,str_modelSum)
ax1.axis('off')



ax1_1 = plt.subplot2grid((10,4),(0,2), colspan=2, rowspan=2)
ax1_1.text(0,1,inputInfo)
ax1_1.axis('off')
pltCol = ['Train', 'Test']
pltRow = ['From', 'To', 'Bull/Bear', 'Bull%/Bear%']
ax1_1.axis('off')
cellText = [[ inputInfoTable['trainStartYr'],inputInfoTable['trainEndYr']], [inputInfoTable['trainEndYr']-1, inputInfoTable['testEndYr']], [inputInfoTable['trainBuBeC'], inputInfoTable['testBuBeC']], [inputInfoTable['trainBuBeR'],inputInfoTable['testBuBeR']] ]
the_table = plt.table(cellText=cellText, rowLabels = pltRow, colLabels = pltCol, loc=6)



ax2 = plt.subplot2grid((10,4),(3,0), colspan=2, rowspan=3)
plt.plot(recallList, label='Recall {:.3f}'.format(recallList[-1]))
plt.plot(precisionList, label='Precision {:.3f}'.format(precisionList[-1]))
plt.plot(f1List, label='F1 {:.3f}'.format(f1List[-1]))
plt.plot(accuracyList, label='Accuracy {:.3f}'.format(accuracyList[-1]))
plt.legend()
plt.xlabel('Epoch')
plt.title('Test')


ax3 = plt.subplot2grid((10,4),(3,2), colspan=2, rowspan=3)
plt.plot(train_recallList, label='Recall {:.3f}'.format(train_recallList[-1]))
plt.plot(train_precisionList, label='Precision {:.3f}'.format(train_precisionList[-1]))
plt.plot(train_f1List, label='F1 {:.3f}'.format(train_f1List[-1]))
plt.plot(train_accuracyList, label='Accuracy {:.3f}'.format(train_accuracyList[-1]))
plt.legend()
plt.xlabel('Epoch')
plt.title('Train_Val')


ax4 = plt.subplot2grid((10,4),(7,0), colspan=2, rowspan=3)
plt.plot(lossList, label='Test Loss {:.3f}'.format(lossList[-1]))
plt.plot(trainLossList, label='Train Loss {:.3f}'.format(trainLossList[-1]))
plt.plot(trainValLossList, label='Train_Val Loss {:.3f}'.format(trainValLossList[-1]))
plt.legend()
plt.xlabel('Epoch')
plt.grid(True)
plt.title('Loss')



