import matplotlib.pyplot as plt
import numpy as np

#產生圖片
img = np.zeros([255,255])
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if j == 5*i-35 or j == 0.5*i+30 or j == -1.5*i+300 :#or np.random.randint(100, size=1) == 0: #加入噪音點
            continue
        else:
            img[i, j] = 255
#初始 random a,b
a = 100*np.random.rand(3)-50  #a = +-50
b = 1000*np.random.rand(3)-500 #b = +-500
a_b = np.vstack([a,b])         

#save data point
point = []
for i in range(img.shape[0]):
  for j in range(img.shape[1]):
    if img[i,j]==0:
      point.append([i,j])

#function 
def put_value(x,y,k):#input (A_x1,A_y1,class_A1)
  for i in range(len(x)):
    x[i,0] = k[i][0] #取class_a1 的x座標, 放到A_x1 column=0 
    y[i] = k[i][1]   #取class_a1 的y座標, 放到A_y1

def new_a_b(x,y,count):#input (A_x1,A_y1,1)
  x_t = x.T
  x_dot = np.dot(x_t,x)
  try:
    inv = np.linalg.inv(x_dot)
  except np.linalg.LinAlgError: #若無法求出inverse,則回傳原本a,b
    if count==1:
        same_a_b = np.zeros((2,1))
        same_a_b[0,0] = a_b[0,0]
        same_a_b[1,0] = a_b[1,0]
        return same_a_b
    elif count==2:
        same_a_b = np.zeros((2,1))
        same_a_b[0,0] = a_b[0,1]
        same_a_b[1,0] = a_b[1,1]
        return same_a_b
    elif count==3:
        same_a_b = np.zeros((2,1))
        same_a_b[0,0] = a_b[0,2]
        same_a_b[1,0] = a_b[1,2]
        return same_a_b

    
  x_dot = np.dot(inv,x_t)
  x_dot = np.dot(x_dot,y)

  return x_dot

for i in range(10):
  #三類資料空間
  class_a1 = []
  class_a2 = []
  class_a3 = []

  #做分類
  for j in range(len(point)):
    distance = abs(a_b[0]*point[j][0]-point[j][1]+a_b[1])/pow((pow(a_b[0],2)+1),0.5)
    min_value = min(distance)
  #距離線最近的點,歸於該類
    if distance[0]==min_value:
      class_a1.append(point[j])
    elif distance[1]==min_value:
      class_a2.append(point[j])
    elif distance[2]==min_value:
      class_a3.append(point[j])
 
  #更新(a,b)
  #建立 N*2,N*1 矩陣
  A_x1 = np.ones((len(class_a1),2))
  A_x2 = np.ones((len(class_a2),2))
  A_x3 = np.ones((len(class_a3),2))
  A_y1 = np.ones((len(class_a1),1))
  A_y2 = np.ones((len(class_a2),1))
  A_y3 = np.ones((len(class_a3),1))
  put_value(A_x1,A_y1,class_a1)
  put_value(A_x2,A_y2,class_a2)
  put_value(A_x3,A_y3,class_a3)
  new_a_b1 = new_a_b(A_x1,A_y1,1)
  new_a_b2 = new_a_b(A_x2,A_y2,2)
  new_a_b3 = new_a_b(A_x3,A_y3,3)
  a_b = np.hstack([new_a_b1,new_a_b2,new_a_b3])

plt.figure(figsize=(7,7))
x = np.linspace(0,255,num=100)
#j == 5*i-35 or j == 0.5*+30 or j == -1.5*i+300
y1 = 5*x-35
y2 = 0.5*x+30
y3 = -1.5*x+300
kmeans_y1 = a_b[0,0]*x+a_b[1,0]
kmeans_y2 = a_b[0,1]*x+a_b[1,1]
kmeans_y3 = a_b[0,2]*x+a_b[1,2]

plt.plot(x,y1,color='black')
plt.plot(x,y2,color='black')
plt.plot(x,y3,color='black')
plt.plot(x,kmeans_y1,color='red')
plt.plot(x,kmeans_y2,color='red')
plt.plot(x,kmeans_y3,color='red')
plt.show()