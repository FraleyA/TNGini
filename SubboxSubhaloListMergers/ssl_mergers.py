import sys
sys.path.append('/home/fraley.a/packages')

import h5py
import numpy as np
import illustris_python as il


def merger_candidates(base_path, subboxNum=0, SSLSnapNum=99, minGas=10, minDM=10, minStar=10, minBH=0):
    
    # Get subhalo data for full-box at the snapshot of the subbox subhalo list file for minimum particle criteria
    FullboxSSLSnapLenType = il.groupcat.loadSubhalos(base_path, SSLSnapNum, fields=['SubhaloLenType'])
    Ngas = FullboxSSLSnapLenType[:, 0]
    Ndm = FullboxSSLSnapLenType[:, 1]
    Nstar = FullboxSSLSnapLenType[:, 4]
    Nbh = FullboxSSLSnapLenType[:, 5]
    
    # Subbox subhalo list file
    subbox_dir = base_path + '/postprocessing/SubboxSubhaloList'
    subbox_file = subbox_dir + f'/subbox{subboxNum}_{SSLSnapNum}.hdf5'
    subfind_ids = h5py.File(subbox_file, 'r')['SubhaloIDs']
    
    # Indices of subhalos which meet the minimum particles criteria
    candidates = np.where((Ngas >= minGas) & (Ndm >= minDM) & (Nstar >= minStar) & (Nbh >= minBH))[0]
    subfind_candidates = np.intersect1d(candidates, subfind_ids)
    
    initial_num = len(subfind_ids)
    filtered_num = len(subfind_candidates)
    print(f'Merger candidates narrowed down from {initial_num} to {filtered_num}.')
    
    return subfind_candidates
    
    
def tree_traversal(base_path, SSLSubfindID, subboxNum=0, SSLSnapNum=99, minMassRatio=0.1, index=0, h=0.6774):
    """ Walk through a merger tree given the subhalo's subfind ID as input.
        base_path: path to simulation data, i.e. .../TNG100-1/output
        SSLSubfindID: SubfindIDs for each subhalo contained in the 'SubhaloID' field
        SSLSnapNum: default 99, walking backwards through progenitor IDs to find mergers
    """
    
    # Load the tree data
    fields = ['SnapNum', 'SubhaloID', 'SubfindID', 'FirstProgenitorID', 'NextProgenitorID', 'DescendantID', 'MainLeafProgenitorID', 'SubhaloMassType']
    tree = il.sublink.loadTree(base_path, SSLSnapNum, SSLSubfindID, fields=fields)

    # Count the number of mergers
    mrgNum = 0

    # Define the inverse of the minimum mass ratio
    invMassRatio = 1 / minMassRatio
    
    # Subbox subhalo list file
    subbox_dir = base_path + '/postprocessing/SubboxSubhaloList'
    subbox_file = subbox_dir + f'/subbox{subboxNum}_{SSLSnapNum}.hdf5'
    subfind_ids = h5py.File(subbox_file, 'r')['SubhaloIDs'] # Really the subfindID of each respective subhalo
    
    # Locate the current_id's index in the subbox subhalo list
    subbox_index = np.where(subfind_ids[:] == SSLSubfindID)[0][0]
    subhaloMinSBSnap = h5py.File(subbox_file, 'r')['SubhaloMinSBSnap'][subbox_index]
    subhaloMaxSBSnap = h5py.File(subbox_file, 'r')['SubhaloMaxSBSnap'][subbox_index]
    subboxSnapNum = h5py.File(subbox_file, 'r')['SubboxSnapNum']
    
    # Initialize subbox scale factor array to get the redshift of corresponding subbox snapshots
    subboxScaleFac = h5py.File(subbox_file, 'r')['SubboxScaleFac']
    
    # Initialize SSLSubfindID dataset
    data = {SSLSubfindID: {
                           'FirstProgenitorFullboxSnapNums': np.array([], dtype=int), 'FirstProgenitorSubboxSnapNums': np.array([], dtype=int), 'FirstProgenitorIDs': np.array([], dtype=int), 'FirstProgenitorStellarMasses': np.array([], dtype=float), 'FirstProgenitorRedshift': np.array([], dtype=float),
                           'NextProgenitorFullboxSnapNums': np.array([], dtype=int), 'NextProgenitorSubboxSnapNums': np.array([], dtype=int), 'NextProgenitorIDs': np.array([], dtype=int), 'NextProgenitorStellarMasses': np.array([], dtype=float), 'NextProgenitorRedshift': np.array([], dtype=float),
                           'DescendantFullboxSnapNums': np.array([], dtype=int), 'DescendantSubboxSnapNums': np.array([], dtype=int), 'DescendantIDs': np.array([], dtype=int), 'DescendantStellarMasses': np.array([], dtype=float), 'DescendantRedshift': np.array([], dtype=float), 'DescendantNumBHs': np.array([], dtype=int),
                           'MassRatios': np.array([], dtype=float)
                          }
           }

    # Store relevant values at the current index
    rootID = tree['SubhaloID'][index]
    rootSubfindID = tree['SubfindID'][index]
    rootSnapNum = tree['SnapNum'][index]
    rootSubboxSnapNum = subboxSnapNum[rootSnapNum] # added this fixing condition in while loop
    firstProgID = tree['FirstProgenitorID'][index]
    firstProgSnapNum = rootSnapNum
    firstProgSubboxSnapNum = rootSubboxSnapNum
    firstProgScaleFac = subboxScaleFac[firstProgSubboxSnapNum]
    firstProgRedshift = (1 / firstProgScaleFac) - 1
   
    # Walk through the tree looking at first progenitors
    while firstProgID != -1 and subhaloMinSBSnap <= firstProgSubboxSnapNum <= subhaloMaxSBSnap:

        # Locate the index where SubhaloID == firstProgID
        firstProgIndex = index + (firstProgID - rootID)

        # Look at the current subhalo's SnapNum and SubboxSnapNum
        firstProgSnapNum = tree['SnapNum'][firstProgIndex]
        firstProgSubboxSnapNum = subboxSnapNum[firstProgSnapNum]
        firstProgScaleFac = subboxScaleFac[firstProgSubboxSnapNum]
        firstProgRedshift = (1 / firstProgScaleFac) - 1
        
        firstProgSubhaloID = tree['SubhaloID'][firstProgIndex]
        firstProgSubfindID = tree['SubfindID'][firstProgIndex]
        
        # Next progenitor pointer
        nextProgID = tree['NextProgenitorID'][firstProgIndex]
        nextProgSnapNum = firstProgSnapNum
        nextProgSubboxSnapNum = firstProgSubboxSnapNum
            
        # This nested loop checks for multiple next progenitors
        while nextProgID != -1:
            
            # Find the next progenitor if it exists and its associated IDs
            nextProgIndex = index + (nextProgID - rootID)
            nextProgSnapNum = tree['SnapNum'][nextProgIndex]
            nextProgSubboxSnapNum = subboxSnapNum[nextProgSnapNum]
            nextProgScaleFac = subboxScaleFac[nextProgSubboxSnapNum]
            nextProgRedshift = (1 / nextProgScaleFac) - 1

            nextProgSubhaloID = tree['SubhaloID'][nextProgIndex]
            nextProgSubfindID = tree['SubfindID'][nextProgIndex]

            # Look at the descendant and record info about it
            descendantID = tree['DescendantID'][nextProgIndex]
            descendantIndex = index + (descendantID - rootID)
            descendantSnapNum = tree['SnapNum'][descendantIndex]
            descendantSubboxSnapNum = subboxSnapNum[descendantSnapNum]
            descendantSubboxScaleFac = subboxScaleFac[descendantSubboxSnapNum]
            descendantRedshift = (1 / descendantSubboxScaleFac) - 1
            descendantSubfindID = tree['SubfindID'][descendantIndex]

            # Calculate descendant stellar mass and number of BHs
            descendantNumBH = il.groupcat.loadSingle(base_path, descendantSnapNum, subhaloID=descendantSubfindID)['SubhaloLenType'][5]

            # Grab progenitor masses
            firstProgMaxMass = il.sublink.maxPastMass(tree, firstProgIndex)
            nextProgMaxMass = il.sublink.maxPastMass(tree, nextProgIndex)
            descendantMaxMass = il.sublink.maxPastMass(tree, descendantIndex)

            # Condition to be considered a merger
            if firstProgMaxMass > 0 and nextProgMaxMass > 0:
                ratio = nextProgMaxMass / firstProgMaxMass
                
                # Redefine 'ratio' if ratio > 1, consistent definition of progenitor mass ratio where 0 < ratio < 1.
                if ratio > 1:
                    # First progenitor has the most "massive history" behind it. Not necessarily the most massive subhalo in the merger.
                    ratio = firstProgMaxMass / nextProgMaxMass
                
                if minMassRatio <= ratio <= invMassRatio:
                    print('Merger found...')

                    # current_id updated to firstProgSubfindID of the previous subhalo, i.e. FirstProgID is now equivalent to subhaloID
                    print(f'SnapNum: {firstProgSnapNum} ({firstProgSubboxSnapNum}) --> FirstProgID: {firstProgSubhaloID} ({firstProgSubfindID})')
                    print(f'SnapNum: {nextProgSnapNum} ({nextProgSubboxSnapNum}) --> NextProgID: {nextProgSubhaloID} ({nextProgSubfindID})')
                    print(f'SnapNum: {descendantSnapNum} ({descendantSubboxSnapNum}) --> DescendantID: {descendantID} ({descendantSubfindID})')
                    print(f'NextProg to FirstProg MassRatio: {ratio}\n')

                    # Add a count to mrgNum
                    mrgNum += 1

                    if mrgNum > 0:
                        
                        # Convert firstProg and nextProg max masses to solar mass (physical)
                        firstProgMaxMass = (firstProgMaxMass * 1e10) / h
                        nextProgMaxMass = (nextProgMaxMass * 1e10) / h
                        descendantMaxMass = (descendantMaxMass * 1e10) / h
                        
                        # Add all of the data to the dictionary!
                        data[SSLSubfindID]['FirstProgenitorFullboxSnapNums'] = np.append(data[SSLSubfindID]['FirstProgenitorFullboxSnapNums'], firstProgSnapNum)
                        data[SSLSubfindID]['FirstProgenitorSubboxSnapNums'] = np.append(data[SSLSubfindID]['FirstProgenitorSubboxSnapNums'], firstProgSubboxSnapNum)
                        data[SSLSubfindID]['FirstProgenitorIDs'] = np.append(data[SSLSubfindID]['FirstProgenitorIDs'], firstProgSubfindID) # Remember current_id index is the first progenitor relative to SSLSubfindID
                        data[SSLSubfindID]['FirstProgenitorStellarMasses'] = np.append(data[SSLSubfindID]['FirstProgenitorStellarMasses'], firstProgMaxMass)
                        data[SSLSubfindID]['FirstProgenitorRedshift'] = np.append(data[SSLSubfindID]['FirstProgenitorRedshift'], firstProgRedshift)

                        data[SSLSubfindID]['NextProgenitorFullboxSnapNums'] = np.append(data[SSLSubfindID]['NextProgenitorFullboxSnapNums'], nextProgSnapNum)
                        data[SSLSubfindID]['NextProgenitorSubboxSnapNums'] = np.append(data[SSLSubfindID]['NextProgenitorSubboxSnapNums'], nextProgSubboxSnapNum)
                        data[SSLSubfindID]['NextProgenitorIDs'] = np.append(data[SSLSubfindID]['NextProgenitorIDs'], nextProgSubfindID)
                        data[SSLSubfindID]['NextProgenitorStellarMasses'] = np.append(data[SSLSubfindID]['NextProgenitorStellarMasses'], nextProgMaxMass)
                        data[SSLSubfindID]['NextProgenitorRedshift'] = np.append(data[SSLSubfindID]['NextProgenitorRedshift'], nextProgRedshift)

                        data[SSLSubfindID]['DescendantFullboxSnapNums'] = np.append(data[SSLSubfindID]['DescendantFullboxSnapNums'], descendantSnapNum)
                        data[SSLSubfindID]['DescendantSubboxSnapNums'] = np.append(data[SSLSubfindID]['DescendantSubboxSnapNums'], descendantSubboxSnapNum)
                        data[SSLSubfindID]['DescendantIDs'] = np.append(data[SSLSubfindID]['DescendantIDs'], descendantSubfindID)
                        data[SSLSubfindID]['DescendantStellarMasses'] = np.append(data[SSLSubfindID]['DescendantStellarMasses'], descendantMaxMass)
                        data[SSLSubfindID]['DescendantRedshift'] = np.append(data[SSLSubfindID]['DescendantRedshift'], descendantRedshift)
                        data[SSLSubfindID]['DescendantNumBHs'] = np.append(data[SSLSubfindID]['DescendantNumBHs'], descendantNumBH)

                        data[SSLSubfindID]['MassRatios'] = np.append(data[SSLSubfindID]['MassRatios'], ratio)
            
            # Attempt to find a pointer to a next progenitor link of current ID
            nextProgID = tree['NextProgenitorID'][nextProgIndex]
        
        # Attempt to find a pointer to a first progenitor link of current ID
        firstProgID = tree['FirstProgenitorID'][firstProgIndex]
            
    print(f'Subhalo {SSLSubfindID} in snapshot {SSLSnapNum} underwent {mrgNum} mergers throughout its history.\n')
    return data