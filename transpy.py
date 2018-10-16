import argparse
import sys
import os
import numpy as np
from operator import add
import random
from scipy.optimize import curve_fit

import He_parameters
import Ne_parameters
import Ar_parameters
import N2_parameters

#-------------------------------------------------------------------------------------------------#
# function to read the coordinates of atoms in the given molecule                                 #
#-------------------------------------------------------------------------------------------------#
def read_log(species_xyz):
    with open(species_xyz,'r') as f:
        lines = f.readlines()
        coords = []
        center_number = 0               
        for line in lines:
            if line.strip() != '' and not line.startswith('!'):
                atomic_symbol, x,y,z = filter(None,line.split(' '))
                center_number = center_number + 1
                if atomic_symbol == 'H' or atomic_symbol == 'h': 
                    atomic_number = '1'
                if atomic_symbol == 'C' or atomic_symbol == 'c': 
                    atomic_number = '6'
                if atomic_symbol == 'N' or atomic_symbol == 'n': 
                    atomic_number = '7'
                if atomic_symbol == 'O' or atomic_symbol == 'o': 
                    atomic_number = '8'
                    
                x = float(x)
                y = float(y)
                z = float(z)
                coords.append([center_number, atomic_number, x,y,z])      
            elif line.startswith('!') or line.strip() == '':
                pass
            else:
                raise Exception("Invalid coordinate file. Please follow *.xyx format.")
    return coords

#-------------------------------------------------------------------------------------------------#
# function to calculate the COM of the given molecule                                             #
#-------------------------------------------------------------------------------------------------#    
def COM_calculator(coords):
    MW = 0
    xcm = 0
    ycm = 0
    zcm = 0

    for atom in coords:
        center_number = atom[0]
        atomic_number = atom[1]
        x = atom[2]
        y = atom[3]
        z = atom[4]
                    
        if atomic_number == '1':
            MW = MW + 1.0
            xcm = xcm + 1.0*x
            ycm = ycm + 1.0*y
            zcm = zcm + 1.0*z
        if atomic_number == '6':
            MW = MW + 12.0
            xcm = xcm + 12.0*x
            ycm = ycm + 12.0*y
            zcm = zcm + 12.0*z
        if atomic_number == '7':
            MW = MW + 14.0
            xcm = xcm + 14.0*x
            ycm = ycm + 14.0*y
            zcm = zcm + 14.0*z
        if atomic_number == '8':
            MW = MW + 16.0
            xcm = xcm + 16.0*x
            ycm = ycm + 16.0*y
            zcm = zcm + 16.0*z
            
    xcm = float(xcm)/MW
    ycm = float(ycm)/MW
    zcm = float(zcm)/MW
         
    return [xcm,ycm,zcm]

#-------------------------------------------------------------------------------------------------#
# function to obtain N uniform orientations around the molecule                                   #
#-------------------------------------------------------------------------------------------------#
def fibonacci_sphere(samples=1):
    rnd = 1.
    if randomize:
        rnd = random.random()*samples

    points = []
    offset = 2./samples
    increment = np.pi * (3. - np.sqrt(5.));

    for i in range(samples):
        y = ((i * offset) - 1) + (offset / 2);
        r = np.sqrt(1 - pow(y,2))

        phi = ((i + rnd) % samples)*increment

        x = np.cos(phi) * r
        z = np.sin(phi) * r

        points.append([x,y,z])

    return points
   
#-------------------------------------------------------------------------------------------------#
# function to generate grid around the given molecule for each orientation or direction           #
#-------------------------------------------------------------------------------------------------#     
def generate_grid(coords,direction):
    COM = COM_calculator(coords)

    # begin at a distance of rCOM_min from COM
    t = np.sqrt(rCOM_min**2/((COM[0]-direction[0])**2+(COM[1]-direction[1])**2+(COM[2]-direction[2])**2))
    point1 = [t*i for i in direction]
    
    rij = []
    for atom in coords:
        distance = np.sqrt((atom[2]-point1[0])**2 + (atom[3]-point1[1])**2 + (atom[4]-point1[2])**2)
        rij.append(distance)
    
    increment = 0.1
    while min(rij) < 1.5:
        r_COM = 2.0 + increment
        t = np.sqrt(r_COM**2/((COM[0]-direction[0])**2+(COM[1]-direction[1])**2+(COM[2]-direction[2])**2))
        point1 = [t*i for i in direction] 
        
        rij = []
        for atom in coords:
            distance = np.sqrt((atom[2]-point1[0])**2 + (atom[3]-point1[1])**2 + (atom[4]-point1[2])**2)
            rij.append(distance)
        
        increment = increment + 0.2
        
    r1 = np.sqrt((COM[0]-point1[0])**2 + (COM[1]-point1[1])**2 + (COM[2]-point1[2])**2)
    
    distances = np.linspace(r1,r1+rCOM_int,rCOM_N)
    grid = []
    for distance in distances:
        t = np.sqrt(distance**2/((COM[0]-direction[0])**2+(COM[1]-direction[1])**2+(COM[2]-direction[2])**2))
        grid_point = [t*i for i in direction]
        grid.append(grid_point)
        
    return grid

#-------------------------------------------------------------------------------------------------#
# Step:0 Define constants                                                                         #
#-------------------------------------------------------------------------------------------------#
N = 2000                        # Number of orientations or directions of approach of bath gas
randomize = False               # Flag for random vs fixed orientations  
hartrees2cm = 2.1947e5          # conversion factor for hartrees to cm-1
cm2kJmol = 1.1963e-2            # conversion factor for cm-1 to kJ/mol
kB = 1.3806e-23                 # Boltzmann constant J/K
N_ava = 6.022e23                # Avagadro's number
cm2K = cm2kJmol*1000/kB/N_ava   # conversion factor for well depth from cm-1 to Kelvin (epsilon/kB)
rCOM_min = 2.0                  # Minimum allowable COM distance in Angstrom for A--Bath 
rCOM_int = 5.0                  # Range of allowable COM distance in Angstrom for A--Bath
rCOM_N = 51                     # Number of points along each one dimensional PES

#-------------------------------------------------------------------------------------------------#
# Step:1 Read input arguments, obtain coordinates and translate the center of mass to origin      #
#-------------------------------------------------------------------------------------------------#    
parser = argparse.ArgumentParser(description="Calculate Lennard-Jones parameters for any CHNO molecule",\
                                 epilog = "Example: python transpy.py CH4.xyz Ne GM")

parser.add_argument("Input_file",help="*.xyz file containing the coordinates of the molecule", type=str)
#parser.add_argument("Bath_gas",help="Choose bath gas from [He, Ne, Ar, N2]", type=str)
#parser.add_argument("Combining_rule",help="Choose combining rule: GM for geometric mean, HM for harmonic mean", type=str)

args = parser.parse_args()

species_xyz = args.Input_file
species_name = species_xyz.split('.')[0]
#bath_gas = args.Bath_gas
#if bath_gas not in ['He','Ne','Ar','N2']:
#    raise ValueError("Invalid bath gas (Please select a bath gas from He, Ne, Ar, N2)")

#combining_rule = args.Combining_rule
#if combining_rule not in ['GM','HM','SPM']:
#    raise ValueError("Invalid combining rule (Please select a combining rule from GM (for geometric mean) or HM (for harmonic mean) or SPM (for sixth power mean))")

coords = read_log(species_xyz)
xcm,ycm,zcm = COM_calculator(coords)

for atom in coords:
        atom[2] = atom[2] - xcm
        atom[3] = atom[3] - ycm
        atom[4] = atom[4] - zcm

with open('Lennard-Jones-parameters.txt','w') as File:
    File.writelines("%20s %15s\n" %('Species name: ',species_name))
    File.writelines("--------------------------------------------------------------\n")

for bath_gas in ['He','Ne','Ar','N2']:
    if bath_gas == 'He':
        parameters = He_parameters
    elif bath_gas == 'Ne':
        parameters = Ne_parameters
    elif bath_gas == 'Ar':
        parameters = Ar_parameters
    elif bath_gas == 'N2':
        parameters = N2_parameters        

    AC = parameters.AC 
    BC = parameters.BC
    CC = parameters.CC
    DC = parameters.DC
    AH = parameters.AH
    BH = parameters.BH
    CH = parameters.CH
    DH = parameters.DH
    AN = parameters.AN
    BN = parameters.BN
    CN = parameters.CN
    DN = parameters.DN
    AO = parameters.AO
    BO = parameters.BO
    CO = parameters.CO
    DO = parameters.DO

    #-----------------------------------------------------------------------------------------------------------#
    # Step:2 Generate N uniform directions around the molecule and obtain one dimentional PES in each direction #
    #-----------------------------------------------------------------------------------------------------------#
    directions = fibonacci_sphere(N)
    #directions = [[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1],[1,1,1],[1,1,-1],[1,-1,1],[1,-1,-1],[-1,1,1],[-1,1,-1],[-1,-1,1],[-1,-1,-1]]
    PES_total = []
    lj = []
    for direction in directions:
        grid = generate_grid(coords, direction)
        PES = []
        COM = COM_calculator(coords)
        for grid_point in grid:
            en = 0.0
            if bath_gas == 'He' or bath_gas == 'Ne' or bath_gas == 'Ar':
                r_COM = np.sqrt((COM[0]-grid_point[0])**2 + (COM[1]-grid_point[1])**2 + (COM[2]-grid_point[2])**2)
                for atom in coords:
                    Rij = np.sqrt((atom[2]-grid_point[0])**2 + (atom[3]-grid_point[1])**2 + (atom[4]-grid_point[2])**2)
                    if atom[1] == '1':
                        en = en + AH*np.exp(-Rij/BH) - (CH**6)/(Rij**6 + DH**6)
                    if atom[1] == '6':
                        en = en + AC*np.exp(-Rij/BC) - (CC**6)/(Rij**6 + DC**6)
                    if atom[1] == '7':
                        en = en + AN*np.exp(-Rij/BN) - (CN**6)/(Rij**6 + DN**6)
                    if atom[1] == '8':
                        en = en + AO*np.exp(-Rij/BO) - (CO**6)/(Rij**6 + DO**6)
            
            if bath_gas == 'N2' or bath_gas == 'H2':
                if bath_gas == 'N2':
                    bath_bond_len = 1.0975
                if bath_gas == 'H2':
                    bath_bond_len = 0.74
                    
                r_COM = np.sqrt((COM[0]-grid_point[0])**2 + (COM[1]-grid_point[1])**2 + (COM[2]-grid_point[2])**2) + bath_bond_len/2.0
                for atom in coords:
                    Rij = np.sqrt((atom[2]-grid_point[0])**2 + (atom[3]-grid_point[1])**2 + (atom[4]-grid_point[2])**2)
                    if atom[1] == '1':
                        en = en + AH*np.exp(-Rij/BH) - (CH**6)/(Rij**6 + DH**6)
                    if atom[1] == '6':
                        en = en + AC*np.exp(-Rij/BC) - (CC**6)/(Rij**6 + DC**6)
                    if atom[1] == '7':
                        en = en + AN*np.exp(-Rij/BN) - (CN**6)/(Rij**6 + DN**6)
                    if atom[1] == '8':
                        en = en + AO*np.exp(-Rij/BO) - (CO**6)/(Rij**6 + DO**6)
                        
                for atom in coords:
                    t = np.sqrt(bath_bond_len**2/(direction[0]**2+direction[1]**2+direction[2]**2))
                    bath_coord = list(map(add,grid_point,[t*i for i in direction]))
                    
                    Rij = np.sqrt((atom[2]-bath_coord[0])**2 + (atom[3]-bath_coord[1])**2 + (atom[4]-bath_coord[2])**2)
                    if atom[1] == '1':
                        en = en + AH*np.exp(-Rij/BH) - (CH**6)/(Rij**6 + DH**6)
                    if atom[1] == '6':
                        en = en + AC*np.exp(-Rij/BC) - (CC**6)/(Rij**6 + DC**6)
                    if atom[1] == '7':
                        en = en + AN*np.exp(-Rij/BN) - (CN**6)/(Rij**6 + DN**6)
                    if atom[1] == '8':
                        en = en + AO*np.exp(-Rij/BO) - (CO**6)/(Rij**6 + DO**6)
            
            PES.append([r_COM,en])
            
        lj.append(min(PES, key = lambda x: x[1]))
        PES_total.append(PES)

    #-----------------------------------------------------------------------------------------------------------#
    # Step:3 Post process one dimentional PES in each direction to obtain effective lennard jones parameters    #
    #-----------------------------------------------------------------------------------------------------------#   
    sigma = []
    epsilon = []
    file_out = species_name +'_' + bath_gas +'.out'
    with open(file_out,'w') as File:
        File.writelines("%20s %20s %20s\n" %('Orientation no.', 'sigma (A)','epsilon (cm-1)'))    
        for d in lj:
            File.writelines("%20d %20.2f %20.3f\n" %(lj.index(d)+1, (2.0**(-1.0/6.0))*d[0],-1*hartrees2cm*d[1]))
            sigma.append((2.0**(-1.0/6.0))*d[0])
            epsilon.append((-1*hartrees2cm)*d[1])
        
        sigma_ij = sum(sigma)/len(lj)
        epsilon_ij = sum(epsilon)/len(lj)
        File.writelines('--------------------------------------------------------------\n')
        File.writelines("%20s %20.2f %20.2f\n" %('Minimum = ', min(sigma),min(epsilon)))
        File.writelines("%20s %20.2f %20.2f\n" %('Maximun = ', max(sigma),max(epsilon)))
        File.writelines("%20s %20.2f %20.2f\n" %('Average = ', sigma_ij,epsilon_ij))

        std_sigma = np.std(np.asarray(sigma))
        std_epsilon = np.std(np.asarray(epsilon))

        File.writelines("%20s %20.2f %20.2f\n" %('Std dev = ', std_sigma,std_epsilon))
        File.writelines('\n')
        File.writelines("%20s %20.2f %4s %3.2f %-4s\n" %('Sigma = ', sigma_ij, '+/-',std_sigma,'A'))
        File.writelines("%20s %20.2f %4s %3.2f %-4s\n" %('Epsilon = ', epsilon_ij,'+/-',std_epsilon,'cm-1'))
    """
    print '------------------------------------------------'
    print 'LJ parameters for ', species_name, '+', bath_gas  
    print 'Sigma (A) = ', sigma_ij
    print 'Epsilon (cm-1) = ', epsilon_ij
    """
    #-----------------------------------------------------------------------------------------------------------#
    # Step:4 Use combining rules to obtain pure species Lennard-Jones parameters                                #
    #-----------------------------------------------------------------------------------------------------------#   
    # bath gas Lennard-Jones parameters from Jasper and Miller, C&F 161, 101-110 (2014)
    if bath_gas == 'He':
        sigma_bath = 2.576      # A
        epsilon_bath = 7.098    # cm-1
    if bath_gas == 'Ne':
        sigma_bath = 2.749      # A
        epsilon_bath = 24.74    # cm-1
    if bath_gas == 'Ar':
        sigma_bath = 3.330      # A
        epsilon_bath = 94.87    # cm-1
    if bath_gas == 'N2':
        sigma_bath = 3.681      # A
        epsilon_bath = 67.89    # cm-1

    combining_rule = 'Sixth power mean combining rules:'
    sigma_ii = (2.0*(sigma_ij**6.0) - sigma_bath**6.0)**(1.0/6.0)
    epsilon_ii = cm2K*(1.0/epsilon_bath)*((epsilon_ij*(sigma_ii**6.0 + sigma_bath**6.0)/(2*(sigma_ii**3)*(sigma_bath**3)))**2)
    """
    print '------------------------------------------------'
    print 'LJ parameters for pure ', species_name
    print 'Sigma (A) = ', sigma_ii
    print 'Epsilon/kB (K) = ', epsilon_ii
    print '------------------------------------------------'
    """    
    with open('Lennard-Jones-parameters.txt','a') as File:
        File.writelines('Bath gas = %s\n'%(bath_gas))
        File.writelines("%20s %20.2f\n" %('Sigma (A) = ', sigma_ii))
        File.writelines("%20s %20.2f\n" %('Epsilon/kB (K)= ', epsilon_ii))
        File.writelines('--------------------------------------------------------------\n')
    
    
        

# Sixth-power mean combining rule
def sigma(x,sig):
    return np.power(((np.power(x,6.0) + np.power(sig,6.0))/2.0),(1.0/6.0))
    
def epsilon(x,eps):
    sigma_ii = coeff1[0]
    sigma_jj = xsig_data
    return np.sqrt(eps*x)*np.divide((2.0*np.multiply(np.power(sigma_ii,3.0),np.power(sigma_jj,3.0))),(np.power(sigma_ii,6.0) + np.power(sigma_jj,6.0)))
   
cm2kJmol = 1.1963e-2            # conversion factor for cm-1 to kJ/mol
kB = 1.3806e-23                 # Boltzmann constant J/K
N_ava = 6.022e23                # Avagadro's number
cm2K = cm2kJmol*1000/kB/N_ava   # conversion factor for well depth from cm-1 to Kelvin (epsilon/kB)

xsig_data = np.asarray([2.576,2.749,3.330,3.681])
xeps_data = np.asarray([7.098,24.74,94.87,67.89])

ysig_data = []
yeps_data = []
for bath_gas in ['He','Ne', 'Ar', 'N2']:
    with open(species_name+'_'+bath_gas+'.out', "r") as F:
        lines_LJ = F.readlines()
        
        for i in range(len(lines_LJ)):
            if lines_LJ[i].strip().startswith('Sigma') and lines_LJ[i].strip().endswith('A'):
                sig = float(lines_LJ[i].strip().split('=')[-1].split('+/-')[0])
                eps = float(lines_LJ[i+1].strip().split('=')[-1].split('+/-')[0])
                
        ysig_data.append(sig)
        yeps_data.append(eps)

ysig_data = np.asarray(ysig_data)
yeps_data = np.asarray(yeps_data)

coeff1 = curve_fit(sigma,xsig_data,ysig_data)
collision_dia = coeff1[0]

coeff2 = curve_fit(epsilon,xeps_data,yeps_data)
well_depth = cm2K*coeff2[0]    
            
with open('Lennard-Jones-parameters.txt','a') as File:
    File.writelines("\n")
    File.writelines("#############################################################################\n")
    File.writelines("Optimized Lennard-Jones parameters based on He, Ne, Ar and N2 as bath gases:\n")
    File.writelines("%-40s %25.3f\n" %('collision diameter (sigma) in angstrom: ',collision_dia))
    File.writelines("%-40s %25.3f\n" %('well-depth (epsilon/kB) in Kelvin: ',well_depth))
    File.writelines("#############################################################################")
    
           
