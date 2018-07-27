import os
import pandas as pd
import csv
from shutil import copyfile
import shutil


dir_imagelist = 'nus-wide-lite/NUS-WIDE-Lite/image list'
dir_groundtruth = 'nus-wide-lite/NUS-WIDE-Lite/NUS-WIDE-Lite_groundtruth'



#save images to lists
images = list()
types = list()
list_files = list()

_files = os.listdir(dir_imagelist)
for _file in (_files):
 list_files.append(_file)

list_files.sort(reverse=True)

for _file in (list_files):

 _type = _file.split('_')
 _type = _type[0]

 path = os.path.join(dir_imagelist, _file)
 with open(path) as f:
#read file line by line 
  content = f.readlines()
  for x in content:
   images.append(x.strip('\n'))
   types.append(_type)


#create dataframe
df = pd.DataFrame(images, columns=['Images'])
df['Type'] = types


#Load groundtruth data
_files = os.listdir(dir_groundtruth)

_catfiles = []
for _file in _files:
 _catfiles.append(_file)

# sort files so that train and test txt files of the same category to be close each other
_catfiles.sort(reverse=True)

train = list()
test = list()
for _file in _catfiles:
 _train = _file.find('Train')
 if _train==-1:
  test.append(_file)
 else:
  train.append(_file)


for x in range (0,len(test)):
 _category = train[x].split('_')
 _category = _category[2]
 
 T1 = list()
 
 path = os.path.join(dir_groundtruth, train[x])

 f = open(path)
 while True:
  line = f.readline()
  line = line.strip('\n')
  if (line =='1'or line=='0'):
   T1.append(line)
  if not line:
   break
 f.close()

 
 path = os.path.join(dir_groundtruth, test[x])

 f = open(path)
 while True:
  line = f.readline()
  line = line.strip('\n')
  if (line =='1'or line=='0'):
   T1.append(line)
  if not line:
   break
 f.close()

 #convert chars to integers
 T1 = list(map(int, T1))
 
 df[_category] = T1

 
print(df.head(3))

print('Total train images:', len(df[df['Type']=='Train']))
print('Total test images:', len(df[df['Type']=='Test']))

counter = 0
for i in range(0, len(df)):


#don't touch skip records
 if df['Type'][i] == 'Test':
  continue
 else:
  counter +=1
  if counter == 7:
   df.at[i, 'Type'] = 'Validation' 
   counter = 0

counter = 0
for i in range(0, len(df)):
 if df['Type'][i] != 'Validation':
  continue
 else:
  counter +=1
  if (counter ==4):
   df.at[i, 'Type'] = 'Groundtruth'
   counter = 0
   print(df.at[i, 'Images'])

 

print('\n\n')
print('After validation spliting:')
print('Total train images:', len(df[df['Type']=='Train']))
print('Total test images:', len(df[df['Type']=='Test']))
print('Total validation images:', len(df[df['Type']=='Validation']))
print('Totan groundtruth images:', len(df[df['Type']=='Groundtruth']))


df.to_csv('SplitedData.csv', sep='\t')

os.makedirs('nus-wide')
os.makedirs('nus-wide/images')
os.makedirs('nus-wide/images/train')
os.makedirs('nus-wide/images/test')
os.makedirs('nus-wide/images/validation')
os.makedirs('nus-wide/images/groundtruth')

print('Copying...')
for x in range (0, len(df)):
 if df.at[x, 'Type']  == 'Train':
  target = 'nus-wide/images/train'
 elif df.at[x, 'Type']  == 'Test':
  target = 'nus-wide/images/test'
 elif df.at[x, 'Type'] == 'Groundtruth':
  target = 'nus-wide/images/groundtruth'
 else:
  target = 'nus-wide/images/validation'

 source = os.path.join('Flickr', df.at[x, 'Images'].replace('\\', '/'))

# print(source)
# print(target)
 shutil.copy(source, target)
 

print('DONE!') 


 





