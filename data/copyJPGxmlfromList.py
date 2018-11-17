import sys
import os
import shutil

fileList=sys.argv[3]
################copy jpg#################################
fromDir = sys.argv[1]+'-img'
targetDir=sys.argv[2]+'-img'
if not os.path.isdir("./"+targetDir+"/"):
    os.makedirs("./"+targetDir+"/")
f = open(fileList)
line = f.readline()
while line:
    a = line.strip('\n')
    tmp = a.split(' ')
    filename =  os.path.basename(tmp[0])
    print filename
    exists = os.path.exists(os.getcwd()+'/'+fromDir+'/'+filename+'.jpg')
    if exists :
        print 'copy '+filename+' to '+os.getcwd()+'/'+targetDir+'/'+filename
        shutil.copy(fromDir+'/'+filename+'.jpg',targetDir+'/'+filename+'.jpg')
    else:
        print filename +' not exists'
    line = f.readline()
#################copy xml#################################
fromDir = sys.argv[1]+'-xml'
targetDir=sys.argv[2]+'_xml'
if not os.path.isdir("./"+targetDir+"/"):
    os.makedirs("./"+targetDir+"/")
f = open(fileList)
line = f.readline()
while line:
    a = line.strip('\n')
    tmp = a.split(' ')
    filename =  os.path.basename(tmp[0])
    print filename
    exists = os.path.exists(os.getcwd()+'/'+fromDir+'/'+filename+'.xml')
    if exists :
        print 'copy '+filename+' to '+os.getcwd()+'/'+targetDir+'/'+filename
        shutil.copy(fromDir+'/'+filename+'.xml',targetDir+'/'+filename+'.xml')
    else:
        print filename +' not exists'
    line = f.readline()
