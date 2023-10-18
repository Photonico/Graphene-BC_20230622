import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tkinter import Tk
from tkinter import filedialog

#### User selection of the working directory#####
def getDIR():
    chooseDIR=input('Is this script located in your working directory? y/n:')
    if(chooseDIR=='y'):
        DIR=os.getcwd()
        print('Your working directory is: '+DIR)
        return DIR
    else:
        root = Tk()
        root.wm_attributes('-topmost',1)
        root.withdraw()
        DIR=filedialog.askdirectory(initialdir=os.getcwd())
        print('Your working directory is: '+DIR)
        return DIR

#### Checks if the files are located in the working directory
def vaspFileCheck(DIR):
    if(os.path.isfile(os.path.join(DIR,'EIGENVAL'))==False):
        print('Please check that the EIGENVAL file is located in your chosen directory')
        quit()
    if(os.path.isfile(os.path.join(DIR,'KPOINTS'))==False):
        print('Please check that the KPOINTS file is located in your chosen directory')
        quit()
    if(os.path.isfile(os.path.join(DIR,'OUTCAR'))==False):
        print('Please check that the KPOINTS file is located in your chosen directory')
        quit()
    else:
        return -1

#### Returns the locations and limits of the high symmetry lines
def getHighSymLines(DIR):
    f=open(os.path.join(DIR,'KPOINTS'),'r')
    KPOINTS=f.readlines()
    Format='rec'
    f.close()
    if(KPOINTS[2][0]!='L'):
        print('Make sure L is on the third line of you KPOINTS file, you currently have: '+KPOINTS[2])
        quit()
    if(KPOINTS[3][0]=='c' or KPOINTS[3][0]=='k'):
        Format='cart'
    lines=int((len(KPOINTS)-4)/2)
    print('The number of High Symmetry lines is '+str(lines))
    limits=[[] for n in range(lines)]
    index=0
    for i in range(4,len(KPOINTS)):
        if((i%2)==0):
            limits[index].append(KPOINTS[i].split())
        if((i%2)!=0):
            limits[index].append(KPOINTS[i].split())
            index+=1
    return Format,lines,limits

#### Returns the calculated fermi energy from the OUTCAR file
def getFermi(DIR):
    f=open(os.path.join(DIR,'OUTCAR'),'r')
    OUTCAR=f.readlines()
    f.close()
    for i in range(0,len(OUTCAR)):
        if('E-fermi' in OUTCAR[i]):
            fermi=OUTCAR[i].split()[2]
            print('The Fermi energy is: '+str(fermi)+' eV')
            return fermi

#### This converts the reciprocal corrdinates to cartesian coordinates for accurate distance calculation
def RectoCart(kL,DIR):
    Basis=input('Please input your k-space lattice basis format, such as fcc, bcc, hcp, sc: ')
    try:
        str(Basis)
    except ValueError:
        return RectoCart(kL,DIR)
    if(getHighSymLines(DIR)[0]=='rec'):
        if(Basis=='fcc' or Basis=='FCC'):
            B=np.array([[-1,1,1],[1,-1,1],[1,1,-1]])
            print('Your lattice is FCC')
        if(Basis=='bcc' or Basis=='BCC'):
            B=np.array([[0,1,1],[1,0,1],[1,1,0]])
            print('Your lattice is BCC')
        if(Basis=='hcp' or Basis=='HCP'): #It may be worth automating the calculation of lengths 'a' and 'c' later.
            a=float(input('What is the side length of your lattice? '))
            c=float(input('What is the height of HCP lattice? '))
            B=np.array([[1,-1/np.sqrt(3),0],[0,2/np.sqrt(3),0],[0,0,a/c]])
            print('Your lattice is HCP')
        if(Basis=='sc' or Basis=='SC'):
            B=np.array([[1,0,0],[0,1,0],[0,0,1]])
            print('Your lattice is SC')
        
        for k in range(len(kL)):
            kL[k]=(kL[k][0]*B[0]+kL[k][1]*B[1]+kL[k][2]*B[2]).tolist()
            
        return kL
    else:
        return kL

#### Reads, processes and plots the band structure from the EIGENVAL file, also calculating the bandgap energy
def Read_EigenFile(DIR):
    f=open(os.path.join(DIR,'EIGENVAL'),'r')
    EIGEN=f.readlines()
    f.close()
    kpointsNum=int(EIGEN[5].split()[1])
    print('Total of '+str(kpointsNum)+' k-points')
    bandNum=int(EIGEN[5].split()[2])
    HSL=int(getHighSymLines(DIR)[1]) 
    bands=[[] for n in range(kpointsNum)]
    kpoints=[[] for n in range(kpointsNum)]
    index=0
    for i in range(7,len(EIGEN)):
        if(i==7):
            kpoints[index].append(float(EIGEN[i].split()[0]))
            kpoints[index].append(float(EIGEN[i].split()[1]))
            kpoints[index].append(float(EIGEN[i].split()[2]))
        if(i>7 and len(EIGEN[i].split())==4):
            index+=1
            kpoints[index].append(float(EIGEN[i].split()[0]))
            kpoints[index].append(float(EIGEN[i].split()[1]))
            kpoints[index].append(float(EIGEN[i].split()[2]))
        if(i>7 and len(EIGEN[i].split())==2):
            bands[index].append(float(EIGEN[i].split()[1]))

    kpoints=RectoCart(kpoints,DIR)
    print('The length of kpoints is '+str(len(kpoints)))

    #Subtract fermi level:
    bands=np.asarray(bands)
    bands=bands-float(getFermi(DIR))
    bands=bands.tolist()
    
    #Calculate distance between points
    dis=[]
    dis.append(0.0) #Initial length making dis array same length as number of k-points
    for j in range(1,len(kpoints)):
        dis.append(abs(np.sqrt((kpoints[j][0]-kpoints[j-1][0])**2+(kpoints[j][1]-kpoints[j-1][1])**2+(kpoints[j][2]-kpoints[j-1][2])**2)))
    #High Symmetry line distances
    lineDis=np.add.reduceat(dis, np.arange(0, len(dis), int(kpointsNum/HSL)), dtype=float)
    for g in range(1,len(lineDis)):
        lineDis[g]=lineDis[g]+lineDis[g-1]
    lineDis=lineDis.tolist()
    lineDis.insert(0,0.0)
    print(lineDis)
    #k-point distance of all lines, to be utilised as the x-axis
    for k in range(1,len(dis)):
        dis[k]=dis[k]+dis[k-1]
    
    #Find the band gap
    Ldex=[0,0]
    Hdex=[0,0]
    Min=999999
    Max=-999999
    for e in range(len(bands)):
        for f in range(bandNum):
            if(bands[e][f]<=0 and Max<bands[e][f]):
                    Max=bands[e][f]
                    Ldex[0]=e
                    Ldex[1]=f
            if(bands[e][f]>0 and Min>bands[e][f]):
                    Min=bands[e][f]
                    Hdex[0]=e
                    Hdex[1]=f
    #print(Min, Max, Min-Max, Ldex, Hdex)
    bandgap=Min-Max
    print('The band gap of this system is: '+str(bandgap)+' eV')


    #Plot the band structure
    sns.set()
    sns.set_style("darkgrid") #Use Seaborn styles for the graphs
    sns.set_palette(sns.color_palette("hls", int(bandNum))) #Use this line to change the colour palette of the plot
    plt.figure(figsize=(40,10))
    my_xticks = [r'$W$',r'$L$',r'$\Gamma$',r'$X$',r'$W$'] #Need to specify high symmetry lines manually

    for m in range(bandNum):
        plt.xticks(lineDis, my_xticks)
        plt.plot(dis,[band[m] for band in bands])
    for n in range(HSL):
        if((n+1)<HSL):
            plt.axvline(x=dis[int(kpointsNum/HSL*(n+1))],color='k',alpha=0.8)
        if((n+1==HSL)):
            plt.axvline(x=dis[int((kpointsNum/HSL*(n+1))-1)],color='k',alpha=0.8)
    plt.axhline(y=0, color='k', alpha=0.4) #line at the calibrated fermi level
    plt.axhline(y=bands[Hdex[0]][Hdex[1]], xmin=dis[Hdex[0]]/dis[len(dis)-1]-dis[Hdex[0]]/dis[len(dis)-1]*0.1, xmax=dis[Hdex[0]]/dis[len(dis)-1]+dis[Hdex[0]]/dis[len(dis)-1]*0.1, color='k')
    plt.axhline(y=bands[Ldex[0]][Ldex[1]], xmin=dis[Hdex[0]]/dis[len(dis)-1]-dis[Hdex[0]]/dis[len(dis)-1]*0.1, xmax=dis[Hdex[0]]/dis[len(dis)-1]+dis[Hdex[0]]/dis[len(dis)-1]*0.1, color='k')
    arrow_properties = dict(fc='black', ec='black', width=1, headwidth=5, shrink=0.1)
    plt.annotate(r"$E_g= $"+str(round(bandgap,3))+' eV', xy=(dis[Hdex[0]], bands[Hdex[0]][Hdex[1]]), xytext=(dis[Hdex[0]], bands[Hdex[0]][Hdex[1]]+3), arrowprops=arrow_properties, horizontalalignment="center")
    plt.annotate("", xy=(dis[Hdex[0]], bands[Ldex[0]][Ldex[1]]), xytext=(dis[Hdex[0]], bands[Ldex[0]][Ldex[1]]-3), arrowprops=arrow_properties, horizontalalignment="center")
    plt.legend()
    plt.xlabel('k-points distance')
    plt.ylabel('Energy (eV)')
    plt.title('Band Structure')
    plt.xlim(dis[0],dis[len(dis)-1])
    fig=plt.gcf()
    plt.show(block=False)

    #Option to save your plot
    save=input('Would you like to save this plot as a PDF? y/n ')
    if(save=='y'):
        fig.savefig(DIR+'\\'+'Band_Structure_Plot.pdf',dpi=600, format='pdf')
        quit()
    else:
        quit()

#####Run the Code#####
DIR=getDIR()
check=vaspFileCheck(DIR)
if(check==-1):
    Read_EigenFile(DIR)
