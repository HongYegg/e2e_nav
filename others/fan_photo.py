# Writer : Fan Mu
#1.Function: With given angle it can mask a sector in figure
#2.Usage: The function Sector is the target function that satisfies the task, Sector(float angle<from 0 to 360> , float theta<default is 5>), 
# this function return a 128*128 mask matrix with 0,1;  the time complexity is O(n), while n means the pixel we must mask; 
#3.Example: Main function shows an example to use this function Sector
#4. Ps. funtion sectoring is a intermediate funtion which is used by function Sector.
import cv2
import math 
import numpy as np

def sectoring(angle ,angle2, mat):
    # an intermediate function to calculate angle != 90 and 270
    ans=mat
    start_a=angle/180*math.pi
    tan_s=math.tan(start_a)
    end_a=(angle2)/180*math.pi
    tan_e=math.tan(end_a)
    flag=1  # flag=0 means I&IV ; 1 means II & III; 2 means across
    ######################################
    #Specicial Cases:
    #########################################
    if(angle==90):
        #print(tan_s)
        for j in range(64,128):
            start=0
            end_a=(angle2-90)/180*math.pi
            tan_e=math.tan(end_a)
            end=int((j-64)*tan_e) 
            for i in range(64,64+end):
                ans[i][j]=0
    if(angle2==90):
        for j in range(64,128):
            start=0
            end_a=(90-angle)/180*math.pi
            tan_e=math.tan(end_a)
            end=int((j-64)*tan_e) 
            for i in range(64-end,64):
                ans[i][j]=0
    if(angle==270):
        for j in range(0,64):
            start=0
            end_a=(angle2-270)/180*math.pi
            tan_e=math.tan(end_a)
            end=int((64-j)*tan_e) 
            for i in range(64-end,64):
                ans[i][j]=0
    if(angle2==270):
        for j in range(0,64):
            start=0
            end_a=(270-angle)/180*math.pi
            tan_e=math.tan(end_a)
            end=int((64-j)*tan_e) 
            for i in range(64,64+end):
                ans[i][j]=0
    ##############################################
    # Normal cases
    # ##########################################    
    if (angle2) <90 or angle>270 :
        flag=0
    if (angle2)< 270 and angle >90:
        flag=1
    if flag ==0:
        for i in range(0,64):
            
            start=int(64+(64-i)*tan_s) 
            end=int(64+(64-i)*tan_e) 
            if start>=end: 
                break
            if start>128 :
                start=127
            if end>128:
                end=127
            if start<0: 
                start=0
            if end<0 : 
                end=0
            for j in range(start -1,end):
                ans[i][j]=0
    if flag==1: 
        for i in range(64,128):
            
            start=int(64+(64-i)*tan_s) 
            end=int(64+(64-i)*tan_e) 
            if start>128:
               start=127
            if end>128:
                end=127
            if start<0: 
                start=0
            if end<0 : 
                end=0
            for j in range(end ,start):
                ans[i][j]=0
    
    return ans

def Sector(angle, theta=5):
    # function to get a mask that fans the figure
    if angle > 360 or angle <0:
        print("Choose a right angle")
        return
    mat=np.ones((128,128))
    #flag=1  # flag=0 means I&IV ; 1 means II & III; 2 means across
    if (angle+theta) <90 or angle>=270 :
        return sectoring(angle,angle+theta, mat)
       
    if (angle+ theta)< 270 and angle >=90:
        return sectoring(angle,angle+theta,mat)
    
    if angle<90 :
        ans=sectoring(angle, 90, mat)
        ans=sectoring(90,angle+theta,ans)
        return ans
    if angle >90 and angle<270 : 
        ans=sectoring(angle, 270, mat)
        ans=sectoring(270,angle+theta,ans)
        return ans



if __name__ == "__main__" :
    # an example
    ###########################
    # get orginal figure
    #############################
    img_path="/home/fanmu/Project/Prj1/Codes/myCodes/test.png"
    org_img=cv2.imread(img_path)
    cv2.imshow("org", org_img)
    #mask
    res=org_img
    mat=Sector(47.2,5)
    for s in range(0,3):
        for i in range (0,127):
            for j in range (0,127):
                res[i][j][s]=org_img[i][j][s]*mat[i][j]
    #get result
    cv2.imshow("test", res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()