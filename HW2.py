import json
import math
from PIL import Image

with open('./Teapot.json') as json_file:
    data = json.load(json_file)
w, h = 700, 700
img = Image.new('RGB',(w, h), (255,255,255))

data1 = data.get('data')
print('dataaaa>>',data)
print('dataaa0>>>',data1[0])
dat = data1[0]
xmin = math.floor(min(data1[0]['v0']['v'][0],data1[0]['v1']['v'][0],data1[0]['v2']['v'][0]))
xmax = math.ceil(max(data1[0]['v0']['v'][0],data1[0]['v1']['v'][0],data1[0]['v2']['v'][0]))
ymin = math.floor(min(data1[0]['v0']['v'][1],data1[0]['v1']['v'][1],data1[0]['v2']['v'][1]))
ymax = math.ceil(max(data1[0]['v0']['v'][1],data1[0]['v1']['v'][1],data1[0]['v2']['v'][1]))
print('coords>>',xmin,xmax,ymin,ymax)

for y in range(ymin,ymax):
  for x in range(xmin, xmax):
    x0 = dat['v0']['v'][0]
    x1 = dat['v1']['v'][0]
    x2 = dat['v2']['v'][0]
    y0 = dat['v0']['v'][1]
    y1 = dat['v1']['v'][1]
    y2 = dat['v2']['v'][1]
    f01 = (y0-y1)*x + (x0-x1)*y + x0*y1 - x1*y0
    f12 = (y1-y2)*x + (x2-x1)*y + x1*y2 - x2*y1
    f20 = (y2-y0)*x + (x0-x2)*y + x2*y0 - x0*y2
    f01a = (y0-y1)*x2 + (x0-x1)*y2 + x0*y1 - x1*y0
    f12a = (y1-y2)*x0 + (x2-x1)*y0 + x1*y2 - x2*y1
    f20a = (y2-y0)*x1 + (x0-x2)*y1 + x2*y0 - x0*y2
    alpha = f12/f12a
    beta = f20/f20a
    gamma = f01/f01a
    print('alBg',alpha,beta,gamma)
    if(alpha > 0 and beta >=0 and gamma >= 0):
      img.putpixel((x,y),(255,0,0))
display(img)
# for dat in data1:
#   # print(dat)
#   print('dat',dat['v0']['v'][0])
