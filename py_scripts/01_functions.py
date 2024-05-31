# Imports
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, date, time
from string import capwords

# Define my_date()
def my_date():
  return datetime.now().strftime('%Y-%m-%d_h%H-m%M-s%S')
my_date()

# Define autoplots()
def autoplots(d, y, line = False):
  '''a function to make a ton of graphs.
  Each plot is based on a subset of d where all variables in the
  plot have no null values.  The size of this subset (n) is 
  displayed in the subtitle of the plot, and can be used 
  similarly to d.isnull().sum(), if desired.
  
  args:
    d: dataframe, the dataframe of the information
    y: string, the name of the column within the dataframe that is the target
    line: bool, whether to plot line graphs, default = False
  return:
    a ton of plots in a FOLDER
  raise:
    pls no'''
  
  # Need these
  from string import capwords
  import matplotlib.pyplot as plt
  import seaborn as sns
  
  # Make a folder 
  try: 
    os.mkdir('images')
  except:
    pass
  a = f'images/plots_{my_date()}'
  os.mkdir(a)
  
  # Define this once
  n_grand = len(d[y])
  print(n_grand)
  # in future versions, I'd like to raise a warning
  # if len(d[y])!=len(d[d[y].notna()])    
  # (i.e., if there are nulls in the target)
  
  # Give y a good(ish) name
  ty = capwords(y.replace('_', ' '))
  print(ty)
  
  # Plot the distributions of all variables
  for i in d.columns:
    print(i)
    # Give it a good(ish) name
    t = capwords(i.replace('_', ' '))
    print(t)
    
    # Extract the subset dataframe, drop NAs, get n
    df = d[i]
    print(df.shape)
    df.dropna(inplace = True)
    n = len(df)
    
    # Plot a histogram of it
    plt.figure(figsize = (16, 9));
    plt.hist(df, bins = 'auto', color = 'purple');
    plt.suptitle(f'Distribution of {t}', size = 24)
    plt.title(f'Based on {n} Observations out of {n_grand}', size = 18)
    plt.xlabel(f'{t}', size = 20);
    plt.ylabel('Frequency', size = 20);
    plt.xticks(size = 16, rotation = 60);
    plt.yticks(size = 16)
    #plt.tight_layout()
    plt.savefig(f'./{a}/{i}_histogram.png')
    plt.close()
    
    # Plot a boxplot of it
    plt.figure(figsize = (16, 9))
    sns.boxplot(data = df, color = 'purple', orient = 'h')
    plt.suptitle(f'Distribution of {t}', size = 24)
    plt.title(f'Based on {n} Observations out of {n_grand}', size = 18)
    plt.xlabel(f'{t}', size = 20);
    plt.xticks(size = 16, rotation = 60)
    #plt.tight_layout()
    plt.savefig(f'./{a}/{i}_boxplot.png')
    plt.close()
    
  # Drop y from the list
  X = [col for col in list(d.drop(columns = [y]).columns)]
  
  # Make plots of each x against y
  for i in X:
    # Give it a good(ish) name
    t = capwords(i.replace('_', ' '))
    print(t)
    
    # Extract the subset dataframe, drop NAs, get n
    df = d[[i, y]]
    df.dropna(inplace = True)
    n = len(df[y])
    
    # Plot a scatterplot of it against y
    plt.figure(figsize = (16, 9))
    plt.scatter(df[i], df[y], alpha = 0.5, color = 'purple')
    plt.suptitle(f'Relationship between {t} and {ty}', size = 24)
    plt.title(f'Based on {n} Observations out of {n_grand}', size = 18)
    plt.xlabel(f'{t}', size = 20);
    plt.ylabel(f'{ty}', size = 20);
    plt.xticks(size = 16, rotation = 60)
    plt.yticks(size = 16)
    # plt.tight_layout()
    plt.savefig(f'./{a}/{t}-by-{y}_scatterplot.png')
    plt.close()
    
    # Plot a line plot of it against y
    if line==True:
      plt.figure(figsize = (16, 9))
      plt.plot(i, y, data = df, color = 'purple')
      plt.suptitle(f'Relationship between {t} and {ty}', size = 24)
      plt.title(f'Based on {n} Observations out of {n_grand}', size = 18)
      plt.xlabel(f'{t}', size = 20);
      plt.ylabel(f'{ty}', size = 20);
      plt.xticks(size = 16, rotation = 60)
      plt.yticks(size = 16)
      # plt.tight_layout()
      plt.savefig(f'./{a}/{i}-by-{y}_lineplot.png')
      plt.close()
    
  # All together now
  n = len(d[y])
  
  # Plot a line plot of everything against y
  if line==True:
    plt.figure(figsize = (16, 9))
    for i in X:
      print(i)
      plt.plot(i, y, data = d)
    plt.suptitle(f'Relationship between Predictors and {ty}', size = 24)
    plt.title(f'Based on {n_grand} Observations out of {n_grand}', size = 18)
    plt.xlabel(f'{t}', size = 20);
    plt.ylabel(f'{ty}', size = 20);
    plt.xticks(size = 16, rotation = 60)
    plt.yticks(size = 16)
    plt.legend();
    # plt.tight_layout()
    plt.savefig(f'./{a}/all-by-{y}_lineplot.png')
    plt.close()
  
  # Get some correlations
  corr = round(d.corr(numeric_only = True), 2)
  
  # Plot a heatmap
  mask = np.zeros_like(corr)
  mask[np.triu_indices_from(mask)] = True
  plt.figure(figsize = (16, 9))
  sns.heatmap(corr, square = True, 
    annot = True, cmap = 'coolwarm', mask = mask);
  plt.suptitle(f'Relationships Between Variables', size = 24)
  plt.title(f'Based on {n_grand} Observations out of {n_grand}', size = 18)
  # plt.tight_layout()
  plt.savefig(f'./{a}/all_heatmap.png')
  plt.close()
  
  # Plot a heatmap column on y
  if y in corr:
    plt.figure(figsize = (16, 9))
    sns.heatmap(np.asarray([corr[y].sort_values(ascending = False)]).T, 
      vmin = 0, vmax = 1, annot = True, cmap = 'coolwarm')
    plt.suptitle(f'Relationship between Predictors and {ty}', size = 24)
    plt.title(f'Based on {n_grand} Observations out of {n_grand}', size = 18)
    plt.xlabel(f'{ty}', size = 20)
    plt.yticklabels = True
    # plt.tight_layout()
    plt.savefig(f'./{a}/all-by-{y}_heatmap.png')
    plt.close()
