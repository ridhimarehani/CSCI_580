import json
import math
from PIL import Image
import numpy as np

n, f, r, l, t, b = 0, 0, 0, 0, 0, 0

def find_len(p):
  return math.sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2])

def unitize(p):
  l = find_len(p)
  return [p[0]/l, p[1]/l, p[2]/l]

def cross(pA, pB):
  a1, a2, a3 = pA[0], pA[1], pA[2];
  b1, b2, b3 = pB[0], pB[1], pB[2];
  param1 = a2 * b3 - a3 * b2
  param2 = a3 * b1 - a1 * b3
  param3 = a1 * b2 - a2 * b1
  return [param1, param2, param3]

def createCamMatrix(camR, to):
  # camU = [1, 0, 0]
  # camV = [0, 1, 0]
  # camN = [0, 0, 1]
  # camMatrix = [[0] * 4 for i in range(4)]
  # camMatrix[0] = [camU[0], camU[1], camU[2], -(camR[0]*camU[0]+camR[1]*camU[1]+camR[2]*camU[2])]
  # camMatrix[1] = [camV[0], camV[1], camV[2], -(camR[0]*camV[0]+camR[1]*camV[1]+camR[2]*camV[2])]
  # camMatrix[2] = [camN[0], camN[1], camN[2], -(camR[0]*camN[0]+camR[1]*camN[1]+camR[2]*camN[2])]
  # camMatrix[3] = [0, 0, 0, 1]

  camN = []
  for i in range(3):
    camN.append(camR[i] - to[i])
  camN = unitize(camN)
  camV = [0, 1, 0]
  camU = cross(camV,camN)
  camU = unitize(camU)
  camV = cross(camN,camU)
  camMatrix = [[0] * 4 for i in range(4)]
  camMatrix[0] = [camU[0], camU[1], camU[2], -(camR[0]*camU[0]+camR[1]*camU[1]+camR[2]*camU[2])]
  camMatrix[1] = [camV[0], camV[1], camV[2], -(camR[0]*camV[0]+camR[1]*camV[1]+camR[2]*camV[2])]
  camMatrix[2] = [camN[0], camN[1], camN[2], -(camR[0]*camN[0]+camR[1]*camN[1]+camR[2]*camN[2])]
  camMatrix[3] = [0, 0, 0, 1]
  # print('camMatrix> ',camMatrix)
  return camMatrix

def normalizeVal(normal):
  denominator = normal[0] * 2 + normal[1]*2 + normal[2] * 2
  res = [normal[0] / denominator, normal[1] / denominator, normal[2] / denominator]
  return res

def multiplyMatrixAndPoint(matrix, point):
  c0r0, c1r0, c2r0, c3r0 = matrix[0][0], matrix[0][1], matrix[0][2], matrix[0][3]
  c0r1, c1r1, c2r1, c3r1 = matrix[1][0], matrix[1][1], matrix[1][2], matrix[1][3]
  c0r2, c1r2, c2r2, c3r2 = matrix[2][0], matrix[2][1], matrix[2][2], matrix[2][3]
  c0r3, c1r3, c2r3, c3r3 = matrix[3][0], matrix[3][1], matrix[3][2], matrix[3][3]

  resultX = (point[0] * c0r0) + (point[1] * c1r0) + (point[2] * c2r0) + (point[3] * c3r0)
  resultY = (point[0] * c0r1) + (point[1] * c1r1) + (point[2] * c2r1) + (point[3] * c3r1)
  resultZ = (point[0] * c0r2) + (point[1] * c1r2) + (point[2] * c2r2) + (point[3] * c3r2)
  resultW = (point[0] * c0r3) + (point[1] * c1r3) + (point[2] * c2r3) + (point[3] * c3r3)

  return [resultX, resultY, resultZ, resultW]

def multiplyMatrix(matrix1, matrix2):

  val1 = [(matrix1[0][0] * matrix2[0][0]) + (matrix1[0][1] * matrix2[1][0]) + (matrix1[0][2] * matrix2[2][0]) + (matrix1[0][3] * matrix2[3][0])]
  val2 = [(matrix1[1][0] * matrix2[0][0]) + (matrix1[1][1] * matrix2[1][0]) + (matrix1[1][2] * matrix2[2][0]) + (matrix1[1][3] * matrix2[3][0])]
  val3 = [(matrix1[2][0] * matrix2[0][0]) + (matrix1[2][1] * matrix2[1][0]) + (matrix1[2][2] * matrix2[2][0]) + (matrix1[2][3] * matrix2[3][0])]
  val4 = [(matrix1[3][0] * matrix2[0][0]) + (matrix1[3][1] * matrix2[1][0]) + (matrix1[3][2] * matrix2[2][0]) + (matrix1[3][3] * matrix2[3][0])]
  
  return val1[0], val2[0], val3[0], val4[0]

def calculateTranspose(matrix):
    transposed = [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]
    return transposed

#world-> cam, cam -> NDC
def WSToNDC(camMatrix,NDCMat,vert):
  # pt = [vert[0],vert[1],vert[2], 1]
  pt = [vert[0],vert[1],vert[2], vert[3]]
  camPt = multiplyMatrixAndPoint(camMatrix,pt)
  proj_plane = 3
  doItForReal=1
  if doItForReal == 1:
    NDCPt = multiplyMatrixAndPoint(NDCMat,camPt)
    
    NDCx = NDCPt[0]/NDCPt[3]
    NDCy = NDCPt[1]/NDCPt[3]
    NDCz = NDCPt[2]/NDCPt[3]
    # print('NDCPt> ',NDCx, NDCy, NDCz)
    # print('NDC Points>> ',NDCx,' >',NDCy,' >','> ',NDCz)
  return [NDCx,NDCy]

def get_params(dataScene):
  res = []
  light1_val = [dataScene['scene']['lights'][0]['id'],dataScene['scene']['lights'][0]['type'],dataScene['scene']['lights'][0]['color'],dataScene['scene']['lights'][0]['intensity']]
  #light1: id, type, color, intensity
  light2_val = [dataScene['scene']['lights'][1]['id'],dataScene['scene']['lights'][1]['type'],dataScene['scene']['lights'][1]['color'],dataScene['scene']['lights'][1]['intensity'],dataScene['scene']['lights'][1]['from'],dataScene['scene']['lights'][1]['to']]
  #light2: id, type, color, intensity, from, to
  camera_val = [dataScene['scene']['camera']['from'],dataScene['scene']['camera']['to'],dataScene['scene']['camera']['bounds'],dataScene['scene']['camera']['resolution']]
  #camera: from, to, bounds, resolution
  res = [light1_val,light2_val,camera_val]
  return res

def create_NDC_matrix(cam_bounds):
  # print('camBounds Mat>>',cam_bounds)
  # print('camBounds>>',cam_bounds[4])
  # print('camBounds>>',cam_bounds[5])
  n = cam_bounds[0]
  f = cam_bounds[1]
  r = cam_bounds[2]
  l = cam_bounds[3]
  t = cam_bounds[5]
  b = cam_bounds[4]
  # NDCMat1 = [[2*n/(r-l), 0, (r+l)/(r-l), 0], 
  #         [0, 2*n/(t-b), (t+b)/(t-b), 0], 
  #         [0, 0, -(f+n)/(f-n), -2*f*n/(f-n)], 
  #         [0, 0, -1, 0]]
  # print('NDCMat1',NDCMat1)
  NDCMatrix = [[3, 0, 0, 0], [0, 3, 0, 0], [0, 0, -1.8571428571428572, -8.571428571428571], [0, 0, -1, 0]]
  return NDCMatrix 

def cross_product(matrix,vertex_vals):
  res = [(matrix[0][0]*vertex_vals[0] + matrix[0][1]*vertex_vals[1] + matrix[0][2]*vertex_vals[2] + matrix[0][3]*vertex_vals[3]),
         (matrix[1][0]*vertex_vals[0] + matrix[1][1]*vertex_vals[1] + matrix[1][2]*vertex_vals[2] + matrix[1][3]*vertex_vals[3]),
         (matrix[2][0]*vertex_vals[0] + matrix[2][1]*vertex_vals[1] + matrix[2][2]*vertex_vals[2] + matrix[2][3]*vertex_vals[3]),
         (matrix[3][0]*vertex_vals[0] + matrix[3][1]*vertex_vals[1] + matrix[3][2]*vertex_vals[2] + matrix[3][3]*vertex_vals[3])]
  # print('translate res> ',res)
  return res

def translate_final(vertex, shade_matrix, rotate_matrix, transform_matrix):
  shade_matrix1 = cross_product(shade_matrix,vertex)
  rotate_matrix1 = cross_product(rotate_matrix,shade_matrix1)
  transform_matrix1 = cross_product(transform_matrix,rotate_matrix1)
  # print('rotate_matrix1>> ',rotate_matrix1);
  # print('transform_matrix1>> ',transform_matrix1);
  return transform_matrix1

#TBD
def numpymultiply(mat, val):
  return np.multiply(mat, val)
#Python Script Starts

# fro = [0, 0, 20]
# to = [0, 0, 0]



# dist = 10.0
# n, f, r, l, t, b = 40, 10, dist, -dist, -dist, dist

w, h = 256, 256

with open('./teapot4.json') as json_file:
    data = json.load(json_file)
with open('./TeapotScene.json') as json_file:
    dataScene = json.load(json_file)

data_scene = get_params(dataScene)
fro = data_scene[2][0]
to = data_scene[2][1]
print('fromto>',fro,to)
NDCMat = create_NDC_matrix(data_scene[2][2])
print('NDCMat Final> ',NDCMat)
NDCPt = [[], [], []]
print('data_scene>',data_scene)
print('data_scene0>',data_scene[0])
print('data_scene1>',data_scene[1])
print('data_scene2>',data_scene[2])
resolution = (data_scene[2][3][0],data_scene[2][3][1])
img = Image.new('RGB',resolution, (127,112,96))
z_buffer = [[math.inf] * 512 for i in range(512)]
data1 = data.get('data')
camMatrix = createCamMatrix(fro, to)


dataSceneShapes = dataScene['scene']['shapes']
La = dataScene['scene']['lights'][0]['intensity']
Le = dataScene['scene']['lights'][1]['intensity']
L = dataScene['scene']['lights'][1]['from']
for val in dataSceneShapes:
  Sx, Sy, Sz = val['transforms'][1]['S']
  id = val['id']
  print('teampot ID> ',id)
  notes = val['notes']
  shading_material = val['material']
  geometry = val['geometry']
  Ka = val['material']['Ka']
  Kd = val['material']['Kd']
  Ks = val['material']['Ks']
  n = val['material']['n']
  cs = val['material']['Cs']
  Ry = val['transforms'][0]['Ry']
  S = val['transforms'][1]['S']
  T = val['transforms'][2]['T']
  shading_vals = val['transforms'][1]['S']
  tansform_vals = val['transforms'][2]['T']
  shade_matrix = [[shading_vals[0], 0, 0, 0], [0, shading_vals[1], 0, 0], [0, 0, shading_vals[2], 0], [0, 0, 0, 1]]
  # rotate_matrix = [[1, 0, 0, 0], [0, math.cos(math.radians(Ry)), math.sin(math.radians(Ry))*-1, 0], [0, math.sin(math.radians(Ry)), math.cos(math.radians(Ry)), 0], [0, 0, 0, 1]]
  # rotate_matrix = [[math.cos(math.radians(Ry)), 0, math.sin(math.radians(Ry)), 0], [0, 1, 0, 0], [math.sin(math.radians(Ry))*-1, 0, math.cos(math.radians(Ry)), 0], [0, 0, 0, 1]]
  r = 1
  rotate_by = Ry
  rotate_matrix = [[math.cos(r * rotate_by), 0, math.sin(r * rotate_by), 0], [0, 1, 0, 0],[-math.sin(r * rotate_by), 0, math.cos(r * rotate_by), 0], [0, 0, 0, 1]]
  transform_matrix = [[1, 0, 0, tansform_vals[0]], [0, 1, 0, tansform_vals[1]], [0, 0, 1, tansform_vals[2]], [0, 0, 0, 1]]

  ambient_light = numpymultiply(dataScene['scene']['lights'][0]['color'], La)
  ambient_light = numpymultiply(ambient_light, Ka)
  

  count = 0 
  for dat in data1:
    # count += 1
    x0 = dat['v0']['v'][0]
    x1 = dat['v1']['v'][0]
    x2 = dat['v2']['v'][0]
    y0 = dat['v0']['v'][1]
    y1 = dat['v1']['v'][1]
    y2 = dat['v2']['v'][1]
    z0 = dat['v0']['v'][2]
    z1 = dat['v1']['v'][2]
    z2 = dat['v2']['v'][2]



    vertex_all = [[x0, y0, z0, 1], [x1, y1, z1, 1], [x2, y2, z2, 1]]
    vertex_tranlate_a = translate_final(vertex_all[0], shade_matrix, rotate_matrix, transform_matrix)
    vertex_tranlate_b = translate_final(vertex_all[1], shade_matrix, rotate_matrix, transform_matrix)
    vertex_tranlate_c = translate_final(vertex_all[2], shade_matrix, rotate_matrix, transform_matrix)
    # if count< 5:
    #   print('countvals>',vertex_tranlate_a, vertex_tranlate_b, vertex_tranlate_c)
    #   count += 1
    # print('vertex_all> ',vertex_all)
    # print('vertex_alla> ',vertex_all[0])
    # print('vertex_tranlate_a> ',vertex_tranlate_a)
    
    NDCPt[0] = WSToNDC(camMatrix, NDCMat, vertex_tranlate_a)
    NDCPt[1] = WSToNDC(camMatrix, NDCMat, vertex_tranlate_b)
    NDCPt[2] = WSToNDC(camMatrix, NDCMat, vertex_tranlate_c)

    # print('NDCPt> ',NDCPt)

    scale = 511/2
    x0 = (NDCPt[0][0] + 1) * scale
    x1 = (NDCPt[1][0] + 1) * scale
    x2 = (NDCPt[2][0] + 1) * scale
    y0 = (1- NDCPt[0][1] ) * scale
    y1 = (1- NDCPt[1][1] ) * scale
    y2 = (1- NDCPt[2][1] ) * scale
    # print('points> ',x0, y0, x1, y1, x2, y2)


    # x0 = (x0 + 1) * (511 / 2)
    # y0 = (1 - y0) * (511 / 2)
    # x1 = (x1 + 1) * (511 / 2)
    # y1 = (1 - y1) * (511 / 2)
    # x2 = (x2 + 1) * (511 / 2)
    # y2 = (1 - y2) * (511 / 2)
    
    xmin = math.floor(min(x0, x1, x2))
    xmax = math.ceil(max(x0, x1, x2))
    ymin = math.floor(min(y0, y1, y2))
    ymax = math.ceil(max(y0, y1, y2))

    #Normal Calculation Start
    nx = dat['v0']['n'][0]
    ny = dat['v0']['n'][1]
    nz = dat['v0']['n'][2]
    nx1 = dat['v1']['n'][0]
    ny1 = dat['v1']['n'][1]
    nz1 = dat['v1']['n'][2]
    nx2 = dat['v2']['n'][0]
    ny2 = dat['v2']['n'][1]
    nz2 = dat['v2']['n'][2]
    normal_all = [[[nx], [ny], [nz], [1]], [[nx1], [ny1], [nz1], [1]], [[nx2], [ny2], [nz2], [1]]]
    scale_matrix_normal = [[1 / Sx, 0, 0, 0], [0, 1 / Sy, 0, 0], [0, 0, 1 / Sz, 0], [0, 0, 0, 1]] #TBD change variable- scale_matrix_normal
    nx0, ny0, nz0, nw0 = multiplyMatrix(scale_matrix_normal, normal_all[0]) #Change this code and below 3 lines #Start from here
    nx1, ny1, nz1, nw1 = multiplyMatrix(scale_matrix_normal, normal_all[1])
    nx2, ny2, nz2, nw2 = multiplyMatrix(scale_matrix_normal, normal_all[2])
    # normal_scaled_all = [[[nx0], [ny0], [nz0], [nw1]],[[nx1], [ny1], [nz1], [nw1]],[[nx2], [ny2], [nz2], [nw2]]]
    normal_scaled_all = [[[nx0], [ny0], [nz0], [nw0]],[[nx1], [ny1], [nz1], [nw1]],[[nx2], [ny2], [nz2], [nw2]]]
    scaled_normal_1 = [[nx0], [ny0], [nz0], [nw0]]
    scaled_normal_2 = [[nx1], [ny1], [nz1], [nw1]]
    scaled_normal_3 = [[nx2], [ny2], [nz2], [nw2]]
    
    nx0, ny0, nz0, nw0 = multiplyMatrix(calculateTranspose(rotate_matrix), scaled_normal_1)
    nx1, ny1, nz1, nw1 = multiplyMatrix(calculateTranspose(rotate_matrix), scaled_normal_2)
    nx2, ny2, nz2, nw2 = multiplyMatrix(calculateTranspose(rotate_matrix), scaled_normal_3)
    normal_rotated_all = [[[nx0], [ny0], [nz0], [nw1]],[[nx1], [ny1], [nz1], [nw1]],[[nx2], [ny2], [nz2], [nw2]]]
    nx0, ny0, nz0, nw0 = multiplyMatrix(calculateTranspose(transform_matrix), normal_rotated_all[0])
    nx1, ny1, nz1, nw1 = multiplyMatrix(calculateTranspose(transform_matrix), normal_rotated_all[1])
    nx2, ny2, nz2, nw2 = multiplyMatrix(calculateTranspose(transform_matrix), normal_rotated_all[2])
    normal_translated_all = [[nx0, ny0, nz0, nw0],[nx1, ny1, nz1, nw1], [nx2, ny2, nz2, nw2]]
    # print('transform_matrix> ',transform_matrix)
    # print('normal_rotated_all 0> ',normal_rotated_all[0])
    # print('normal_translated_all 0> ',normal_translated_all[0])
    #Normal Calculation End

    dotp = float(0.707 * nx) + float(0.5 * ny) + float(0.5 * nz)
    # dotp1 = float(0.707 * nx1) + float(0.5 * ny1) + float(0.5 * nz1)
    # dotp2 = float(0.707 * nx2) + float(0.5 * ny2) + float(0.5 * nz2)

    if dotp < 0.0:
      dotp = -dotp
    elif dotp > 1.0:
      dotp = 1.0

    #color Calculation starts
    N = [nx, ny, nz]

    

  
    val3_ambient1 = data_scene[0][3] * Ka * data_scene[0][2][0]
    val3_ambient2 = data_scene[0][3] * Ka * data_scene[0][2][1]
    val3_ambient3 = data_scene[0][3] * Ka * data_scene[0][2][2]
    
   
    rgb = [float(0.95 * dotp), float(0.65 * dotp), float(0.88 * dotp)]
    col = (int(rgb[0] * 255) ,int(rgb[1] * 255), int(rgb[2] * 255))
    #color Calculation ends

    for y in range(ymin,ymax):
      for x in range(xmin, xmax):
        
        f01 = (y0-y1)*x + (x1-x0)*y + x0*y1 - x1*y0
        f12 = (y1-y2)*x + (x2-x1)*y + x1*y2 - x2*y1
        f20 = (y2-y0)*x + (x0-x2)*y + x2*y0 - x0*y2
        f01a = (y0-y1)*x2 + (x1-x0)*y2 + x0*y1 - x1*y0
        f12a = (y1-y2)*x0 + (x2-x1)*y0 + x1*y2 - x2*y1
        f20a = (y2-y0)*x1 + (x0-x2)*y1 + x2*y0 - x0*y2
        if f01a == 0 or f12a == 0 or f20a == 0:
          continue
        alpha = f12/f12a
        beta = f20/f20a
        gamma = f01/f01a

        normal_11 = numpymultiply(normal_translated_all[0], alpha)
        normal_22 = numpymultiply(normal_translated_all[1], beta)
        normal_33 = numpymultiply(normal_translated_all[2], gamma)
        normal_final_all = [numpymultiply(normal_translated_all[0], alpha),numpymultiply(normal_translated_all[1], beta),numpymultiply(normal_translated_all[2], gamma)]
        # normalsFinal = np.add(np.add(n11, n22), n33)
        normalForShading = np.add(np.add(normal_final_all[0], normal_final_all[1]), normal_final_all[2])
        #Shading calculation starts
        normalForShading = normalizeVal(normalForShading);
        R_val = normalizeVal(np.subtract(numpymultiply(normalForShading, 2 * np.dot(normalForShading, L)), L)) 
        ambient_light = numpymultiply(dataScene['scene']['lights'][0]['color'], La)
        ambient_light1 = numpymultiply(ambient_light, Ka)
        diffused_light = Kd * max(np.dot(L, normalForShading), 0)
        diffused_light1 = numpymultiply(diffused_light, dataScene['scene']['lights'][1]['color'])
        specular_light = Ks * math.pow(np.dot(R_val, [0, 0, 0]), n) * Le #TBD
        specular_light1 = numpymultiply(specular_light, dataScene['scene']['lights'][1]['color'])
        ambinet_and_diffused = np.add(ambient_light1, diffused_light1)
        color = numpymultiply(ambinet_and_diffused, cs) #TBD
        color = np.add(color, specular_light1) #TBD
        # print('color> ',color)
        color_final = [int(x * 255) for x in color]
        color_final_tuple = (color_final[0], color_final[1], color_final[2])
        #Shading calculation ends
        z_at_pixel = alpha * z0 + beta * z1 + gamma * z2
        if(alpha >= 0 and beta >=0 and gamma >= 0):
          if(x >= 0 and y >= 0 and x < 512 and y < 512):
            if z_at_pixel < z_buffer[x][y]:
              img.putpixel((x,y),color_final_tuple)
              z_buffer[x][y] = z_at_pixel
display(img)