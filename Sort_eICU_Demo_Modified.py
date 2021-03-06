import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import copy
# import scipy
# import pylab as py
# import matplotlib.pyplot as plt
#import pdvega

import os
import glob

filepath = os.chdir('CSVs')
##a = os.listdir(filepath)
eICU = glob.glob('*.{}'.format('csv'))

# read csv files
for i in range(len(eICU)):
    eICU[i] = pd.read_csv(eICU[i])

### Main tables parameters ####################################################
admissiondrug = []
admissiondx = []
allergy = []
apacheapsvar = []
apachepatientresult = []
apachepredvar = []
cpcareprovider = []
cpeol = []
cpgeneral = []
cpgoal = []
cpinfectiousdisease = []
customlab = []
diagnosis = []
hospital = []
infusiondrug = []
intakeoutput = []
lab = []
medication = []
microlab = []
note = []
nurseassess = []
nursecare = []
nursechart = []
pasthistory = []
physicalexam = []
respiratorycare = []
respiratorychart = []
treatment = []
vitalA = []
vitalP = []

### Table 25: patient sorted based on patienthealthsystemid ###################
patient_num = eICU[24].uniquepid.unique()
patient = []
for j in range(len(patient_num)):
    patient.append(eICU[24][eICU[24].uniquepid == patient_num[j]].sort_values(by = ['unitvisitnumber']))
    patient[j].reset_index(drop=True, inplace=True)    

for k in range(len(patient_num)):
### Table 1: admissiondrug ####################################################
    admissiondrugzeros = np.zeros(len(eICU[0].patientunitstayid))
    for l0 in range(len(patient[k])):
        admissiondrugzeros = admissiondrugzeros + ((eICU[0].patientunitstayid == patient[k].patientunitstayid[l0])*1)

    admissiondrug.append(eICU[0][eICU[0].patientunitstayid == np.multiply(admissiondrugzeros, eICU[0].patientunitstayid)])
    admissiondrug[k].insert(0, 'uniquepid', patient[k].uniquepid[0] ,True)
    
### Table 2: admissiondx ######################################################
    admissiondxzeros = np.zeros(len(eICU[1].patientunitstayid))
    for l1 in range(len(patient[k])):
        admissiondxzeros = admissiondxzeros + ((eICU[1].patientunitstayid == patient[k].patientunitstayid[l1])*1)

    admissiondx.append(eICU[1][eICU[1].patientunitstayid == np.multiply(admissiondxzeros, eICU[1].patientunitstayid)])
    admissiondx[k].insert(0, 'uniquepid', patient[k].uniquepid[0] ,True)
    
### Table 3: allergy ##########################################################
    allergyzeros = np.zeros(len(eICU[2].patientunitstayid))
    for l2 in range(len(patient[k])):
        allergyzeros = allergyzeros + ((eICU[2].patientunitstayid == patient[k].patientunitstayid[l2])*1)

    allergy.append(eICU[2][eICU[2].patientunitstayid == np.multiply(allergyzeros, eICU[2].patientunitstayid)])
    allergy[k].insert(0, 'uniquepid', patient[k].uniquepid[0] ,True)
    
### Table 4: ApacheApsVar #####################################################
    apacheavzeros = np.zeros(len(eICU[3].patientunitstayid))
    for l3 in range(len(patient[k])):
        apacheavzeros = apacheavzeros + ((eICU[3].patientunitstayid == patient[k].patientunitstayid[l3])*1)

    apacheapsvar.append(eICU[3][eICU[3].patientunitstayid == np.multiply(apacheavzeros, eICU[3].patientunitstayid)])
    apacheapsvar[k].insert(0, 'uniquepid', patient[k].uniquepid[0] ,True)

### Table 5: ApachePatientResult ##############################################
    apacheprzeros = np.zeros(len(eICU[4].patientunitstayid))
    for l4 in range(len(patient[k])):
        apacheprzeros = apacheprzeros + ((eICU[4].patientunitstayid == patient[k].patientunitstayid[l4])*1)

    apachepatientresult.append(eICU[4][eICU[4].patientunitstayid == np.multiply(apacheprzeros, eICU[4].patientunitstayid)])
    apachepatientresult[k].insert(0, 'uniquepid', patient[k].uniquepid[0] ,True)

### Table 6: ApachePredVar ####################################################
    apachepvzeros = np.zeros(len(eICU[5].patientunitstayid))
    for l5 in range(len(patient[k])):
        apachepvzeros = apachepvzeros + ((eICU[5].patientunitstayid == patient[k].patientunitstayid[l5])*1)

    apachepredvar.append(eICU[5][eICU[5].patientunitstayid == np.multiply(apachepvzeros, eICU[5].patientunitstayid)])
    apachepredvar[k].insert(0, 'uniquepid', patient[k].uniquepid[0] ,True)

### Table 7: carePlanCareProvider #############################################
    cpcpzeros = np.zeros(len(eICU[6].patientunitstayid))
    for l6 in range(len(patient[k])):
        cpcpzeros = cpcpzeros + ((eICU[6].patientunitstayid == patient[k].patientunitstayid[l6])*1)

    cpcareprovider.append(eICU[6][eICU[6].patientunitstayid == np.multiply(cpcpzeros, eICU[6].patientunitstayid)])
    cpcareprovider[k].insert(0, 'uniquepid', patient[k].uniquepid[0] ,True)

### Table 8: carePlanEOL ######################################################
    cpeolzeros = np.zeros(len(eICU[7].patientunitstayid))
    for l7 in range(len(patient[k])):
        cpeolzeros = cpeolzeros + ((eICU[7].patientunitstayid == patient[k].patientunitstayid[l7])*1)

    cpeol.append(eICU[7][eICU[7].patientunitstayid == np.multiply(cpeolzeros, eICU[7].patientunitstayid)])
    cpeol[k].insert(0, 'uniquepid', patient[k].uniquepid[0] ,True)

### Table 9: carePlanGeneral ##################################################
    cpgenzeros = np.zeros(len(eICU[8].patientunitstayid))
    for l8 in range(len(patient[k])):
        cpgenzeros = cpgenzeros + ((eICU[8].patientunitstayid == patient[k].patientunitstayid[l8])*1)

    cpgeneral.append(eICU[8][eICU[8].patientunitstayid == np.multiply(cpgenzeros, eICU[8].patientunitstayid)])
    cpgeneral[k].insert(0, 'uniquepid', patient[k].uniquepid[0] ,True)

### Table 10: carePlanGoal ####################################################
    cpgoalzeros = np.zeros(len(eICU[9].patientunitstayid))
    for l9 in range(len(patient[k])):
        cpgoalzeros = cpgoalzeros + ((eICU[9].patientunitstayid == patient[k].patientunitstayid[l9])*1)

    cpgoal.append(eICU[9][eICU[9].patientunitstayid == np.multiply(cpgoalzeros, eICU[9].patientunitstayid)])
    cpgoal[k].insert(0, 'uniquepid', patient[k].uniquepid[0] ,True)

### Table 11: carePlanInfectiousDisease #######################################
    cpinfdiseasezeros = np.zeros(len(eICU[10].patientunitstayid))
    for l10 in range(len(patient[k])):
        cpinfdiseasezeros = cpinfdiseasezeros + ((eICU[10].patientunitstayid == patient[k].patientunitstayid[l10])*1)

    cpinfectiousdisease.append(eICU[10][eICU[10].patientunitstayid == np.multiply(cpinfdiseasezeros, eICU[10].patientunitstayid)])
    cpinfectiousdisease[k].insert(0, 'uniquepid', patient[k].uniquepid[0] ,True)

### Table 12: customLab #######################################################
    customlabzeros = np.zeros(len(eICU[11].patientunitstayid))
    for l11 in range(len(patient[k])):
        customlabzeros = customlabzeros + ((eICU[11].patientunitstayid == patient[k].patientunitstayid[l11])*1)

    customlab.append(eICU[11][eICU[11].patientunitstayid == np.multiply(customlabzeros, eICU[11].patientunitstayid)])
    customlab[k].insert(0, 'uniquepid', patient[k].uniquepid[0] ,True)

### Table 13: diagnosis #######################################################
    diagnosiszeros = np.zeros(len(eICU[12].patientunitstayid))
    for l12 in range(len(patient[k])):
        diagnosiszeros = diagnosiszeros + ((eICU[12].patientunitstayid == patient[k].patientunitstayid[l12])*1)

    diagnosis.append(eICU[12][eICU[12].patientunitstayid == np.multiply(diagnosiszeros, eICU[12].patientunitstayid)])
    diagnosis[k].insert(0, 'uniquepid', patient[k].uniquepid[0] ,True)

### Table 14: hospital ########################################################
    hospitalzeros = np.zeros(len(eICU[13].hospitalid))
    #for l13 in range(len(patient[k])):
    hospitalzeros = hospitalzeros + ((eICU[13].hospitalid == patient[k].hospitalid[0])*1)

    hospital.append(eICU[13][eICU[13].hospitalid == np.multiply(hospitalzeros, eICU[13].hospitalid)])
    hospital[k].insert(0, 'uniquepid', patient[k].uniquepid[0], True)

### Table 15: infusionDrug ####################################################
    infusiondrugzeros = np.zeros(len(eICU[14].patientunitstayid))
    for l14 in range(len(patient[k])):
        infusiondrugzeros = infusiondrugzeros + ((eICU[14].patientunitstayid == patient[k].patientunitstayid[l14])*1)

    infusiondrug.append(eICU[14][eICU[14].patientunitstayid == np.multiply(infusiondrugzeros, eICU[14].patientunitstayid)])
    infusiondrug[k].insert(0, 'uniquepid', patient[k].uniquepid[0] ,True)

### Table 16: intakeOutput ####################################################
    intakeoutputzeros = np.zeros(len(eICU[15].patientunitstayid))
    for l15 in range(len(patient[k])):
        intakeoutputzeros = intakeoutputzeros + ((eICU[15].patientunitstayid == patient[k].patientunitstayid[l15])*1)

    intakeoutput.append(eICU[15][eICU[15].patientunitstayid == np.multiply(intakeoutputzeros, eICU[15].patientunitstayid)])
    intakeoutput[k].insert(0, 'uniquepid', patient[k].uniquepid[0], True)

### Table 17: lab #############################################################
    labzeros = np.zeros(len(eICU[16].patientunitstayid))
    for l16 in range(len(patient[k])):
        labzeros = labzeros + ((eICU[16].patientunitstayid == patient[k].patientunitstayid[l16])*1)

    lab.append(eICU[16][eICU[16].patientunitstayid == np.multiply(labzeros, eICU[16].patientunitstayid)])
    lab[k].insert(0, 'uniquepid', patient[k].uniquepid[0], True)

### Table 18: medication ######################################################
    medicationzeros = np.zeros(len(eICU[17].patientunitstayid))
    for l17 in range(len(patient[k])):
        medicationzeros = medicationzeros + ((eICU[17].patientunitstayid == patient[k].patientunitstayid[l17])*1)

    medication.append(eICU[17][eICU[17].patientunitstayid == np.multiply(medicationzeros, eICU[17].patientunitstayid)])
    medication[k].insert(0, 'uniquepid', patient[k].uniquepid[0], True)
    
## len(eICU[17].drugname[(eICU[17].drugname.str.count('metformin') != 0)].dropna())

### Table 19: microLab ########################################################
    microlabzeros = np.zeros(len(eICU[18].patientunitstayid))
    for l18 in range(len(patient[k])):
        microlabzeros = microlabzeros + ((eICU[18].patientunitstayid == patient[k].patientunitstayid[l18])*1)

    microlab.append(eICU[18][eICU[18].patientunitstayid == np.multiply(microlabzeros, eICU[18].patientunitstayid)])
    microlab[k].insert(0, 'uniquepid', patient[k].uniquepid[0], True)

### Table 20: note ############################################################
    notezeros = np.zeros(len(eICU[19].patientunitstayid))
    for l19 in range(len(patient[k])):
        notezeros = notezeros + ((eICU[19].patientunitstayid == patient[k].patientunitstayid[l19])*1)

    note.append(eICU[19][eICU[19].patientunitstayid == np.multiply(notezeros, eICU[19].patientunitstayid)])
    note[k].insert(0, 'uniquepid', patient[k].uniquepid[0], True)

### Table 21: nurseAssessment #################################################
    nurseassesszeros = np.zeros(len(eICU[20].patientunitstayid))
    for l20 in range(len(patient[k])):
        nurseassesszeros = nurseassesszeros + ((eICU[20].patientunitstayid == patient[k].patientunitstayid[l20])*1)

    nurseassess.append(eICU[20][eICU[20].patientunitstayid == np.multiply(nurseassesszeros, eICU[20].patientunitstayid)])
    nurseassess[k].insert(0, 'uniquepid', patient[k].uniquepid[0], True)

### Table 22: nurseCare #######################################################
    nursecarezeros = np.zeros(len(eICU[21].patientunitstayid))
    for l21 in range(len(patient[k])):
        nursecarezeros = nursecarezeros + ((eICU[21].patientunitstayid == patient[k].patientunitstayid[l21])*1)

    nursecare.append(eICU[21][eICU[21].patientunitstayid == np.multiply(nursecarezeros, eICU[21].patientunitstayid)])
    nursecare[k].insert(0, 'uniquepid', patient[k].uniquepid[0], True)

### Table 23: nurseCharting ###################################################
    nursechartzeros = np.zeros(len(eICU[22].patientunitstayid))
    for l22 in range(len(patient[k])):
        nursechartzeros = nursechartzeros + ((eICU[22].patientunitstayid == patient[k].patientunitstayid[l22])*1)

    nursechart.append(eICU[22][eICU[22].patientunitstayid == np.multiply(nursechartzeros, eICU[22].patientunitstayid)])
    nursechart[k].insert(0, 'uniquepid', patient[k].uniquepid[0], True)

### Table 24: pastHistory #####################################################
    pasthistoryzeros = np.zeros(len(eICU[23].patientunitstayid))
    for l23 in range(len(patient[k])):
        pasthistoryzeros = pasthistoryzeros + ((eICU[23].patientunitstayid == patient[k].patientunitstayid[l23])*1)

    pasthistory.append(eICU[23][eICU[23].patientunitstayid == np.multiply(pasthistoryzeros, eICU[23].patientunitstayid)])
    pasthistory[k].insert(0, 'uniquepid', patient[k].uniquepid[0], True)

### Table 26: physicalExam ####################################################
    physicalexamzeros = np.zeros(len(eICU[25].patientunitstayid))
    for l25 in range(len(patient[k])):
        physicalexamzeros = physicalexamzeros + ((eICU[25].patientunitstayid == patient[k].patientunitstayid[l25])*1)

    physicalexam.append(eICU[25][eICU[25].patientunitstayid == np.multiply(physicalexamzeros, eICU[25].patientunitstayid)])
    physicalexam[k].insert(0, 'uniquepid', patient[k].uniquepid[0], True)

### Table 27: respiratoryCare #################################################
    respiratorycarezeros = np.zeros(len(eICU[26].patientunitstayid))
    for l26 in range(len(patient[k])):
        respiratorycarezeros = respiratorycarezeros + ((eICU[26].patientunitstayid == patient[k].patientunitstayid[l26])*1)

    respiratorycare.append(eICU[26][eICU[26].patientunitstayid == np.multiply(respiratorycarezeros, eICU[26].patientunitstayid)])
    respiratorycare[k].insert(0, 'uniquepid', patient[k].uniquepid[0], True)

### Table 28: respiratoryCharting #############################################
    respiratorychartzeros = np.zeros(len(eICU[27].patientunitstayid))
    for l27 in range(len(patient[k])):
        respiratorychartzeros = respiratorychartzeros + ((eICU[27].patientunitstayid == patient[k].patientunitstayid[l27])*1)

    respiratorychart.append(eICU[27][eICU[27].patientunitstayid == np.multiply(respiratorychartzeros, eICU[27].patientunitstayid)])
    respiratorychart[k].insert(0, 'uniquepid', patient[k].uniquepid[0], True)

### Table 29: treatment #######################################################
    treatmentzeros = np.zeros(len(eICU[28].patientunitstayid))
    for l28 in range(len(patient[k])):
        treatmentzeros = treatmentzeros + ((eICU[28].patientunitstayid == patient[k].patientunitstayid[l28])*1)

    treatment.append(eICU[28][eICU[28].patientunitstayid == np.multiply(treatmentzeros, eICU[28].patientunitstayid)])
    treatment[k].insert(0, 'uniquepid', patient[k].uniquepid[0], True)

### Table 30: vitalAperiodic ##################################################
    vitalAzeros = np.zeros(len(eICU[29].patientunitstayid))
    for l29 in range(len(patient[k])):
        vitalAzeros = vitalAzeros + ((eICU[29].patientunitstayid == patient[k].patientunitstayid[l29])*1)

    vitalA.append(eICU[29][eICU[29].patientunitstayid == np.multiply(vitalAzeros, eICU[29].patientunitstayid)])
    vitalA[k].insert(0, 'uniquepid', patient[k].uniquepid[0], True)

### Table 31: vitalPeriodic ###################################################
    vitalPzeros = np.zeros(len(eICU[30].patientunitstayid))
    for l30 in range(len(patient[k])):
        vitalPzeros = vitalPzeros + ((eICU[30].patientunitstayid == patient[k].patientunitstayid[l30])*1)

    vitalP.append(eICU[30][eICU[30].patientunitstayid == np.multiply(vitalPzeros, eICU[30].patientunitstayid)])
    vitalP[k].insert(0, 'uniquepid', patient[k].uniquepid[0], True)

##############################################################################################################################
### Modification of each table based on the descriptions given on https://github.com/MIT-LCP/eicu-code/tree/master/notebooks
##############################################################################################################################

### Table 1: admissiondrug ###################################################################################################

#   768: insulin regular, humulin
#  1661: haldol    
#  4763: glucophage, metformin hcl
# 20769: novolog, novolog flexpen, insulin aspart
#   803: glipizide, glucotrol, glucotrol xl
#   802: glyburide, diabeta
#  9690: glyburide-metformin hcl, glucovance
# 10485: glimepiride, amaryl
#   608: magnesium gluconate
# 11528: insulin lispro
# 22025: insulin glargine, lantus
# 26407: insulin detemir, levemir
# 20334: insulin admin. supplies
# 23038: insulin syringe
#   780: insulin nph, novolin

# 143 patients ---> sample: 1605 ##############################################
drughicllist = (807, 19078, 768, 1661, 525, 6549, 4763, 20769, 803, 802, 9690, 10485, 608, 11528, 22025, 26407, 20334, 23038, 780)
druglist1 = ('metFormin', 'GliPizide', 'GlucoPHaGe', 'GLUcagon', 'inSULIN', 'gLuCotroL', 'glYBUride', 'GluCOVancE', 
            'GLimepIrIdE', 'SUlfOnYlurEA', 'hypoGLYCEMIA', 'Magnesium Gluconate', 'noVOloG', 'aMaRyL', 'LAntUs', 
            'huMALog', 'lEvEmIR', 'HypErGlYCemIa', 'GlUcOsE', 'laCTaTe', 'dIaBeT', 'gLuToSE')
admissiondrug_m = []
y1 = 0

for k in range(len(admissiondrug)):
    diabdrugpat = []
    for x1 in range(len(drughicllist)):
        diabdrugpat.append(admissiondrug[k][(admissiondrug[k].drughiclseqno == drughicllist[x1])])
    
    diabdrugpat = pd.concat(diabdrugpat[:], axis=0, join='outer', ignore_index=False)
    diabdrugpat = diabdrugpat.sort_values(by = ['drugoffset'])
    admissiondrug_m.append(diabdrugpat)

    if len(admissiondrug_m[k]) != 0:
        y1 = y1 + 1
        print('The Corresponding Index is', k, 'with', len(admissiondrug_m[k]), 'sample(s)')
        
print('admissiondrug_m has', y1, 'patients')

### Table 2: admissionDx #####################################################################################################

# 145 patients ---> sample: 179 ###############################################
druglist2 = ['eNdOcRInE']
searchlist = []
admissiondx_m = []
y2 = 0

for k in range(len(admissiondx)):
    diabdrugpat = []
    for x2 in range(len(druglist2)):
        searchlist.append([druglist2[x2].lower(), druglist2[x2].upper(), druglist2[x2].capitalize()])
        diabdrugpat.append(admissiondx[k][(admissiondx[k].admitdxpath.str.count(searchlist[x2][0]) + admissiondx[k].admitdxpath.str.count(searchlist[x2][1]) + admissiondx[k].admitdxpath.str.count(searchlist[x2][2]) == 1)])
    
    diabdrugpat = pd.concat(diabdrugpat[:], axis=0, join='outer', ignore_index=False)
    diabdrugpat = diabdrugpat.sort_values(by = ['admitdxenteredoffset'])
    admissiondx_m.append(diabdrugpat)
    searchlist.clear()

    if len(admissiondx_m[k]) != 0:
        y2 = y2 + 1
        print('The Corresponding Index is', k, 'with', len(admissiondx_m[k]), 'sample(s)')
        
print('admissiondx_m has', y2, 'patients')

#### Table 3: allergy ########################################################################################################

#10 patients ---> sample: 1255, 1043 ##########################################
allergy_m = []
y3 = 0

for k in range(len(allergy)):
    diabdrugpat = []
    diabdrugpatzeros = np.zeros(len(allergy[k]))
    for x3 in range(len(druglist1)):
        searchlist.append([druglist1[x3].lower(), druglist1[x3].upper(), druglist1[x3].capitalize()])
        temp3 = allergy[k].allergyname.str.count(searchlist[x3][0]).fillna(0) + allergy[k].allergyname.str.count(searchlist[x3][1]).fillna(0) + allergy[k].allergyname.str.count(searchlist[x3][2]).fillna(0)
        diabdrugpatzeros = diabdrugpatzeros + temp3
        for x31 in range(len(drughicllist)):
            temp31 = (allergy[k].drughiclseqno == drughicllist[x31])*1
            diabdrugpatzeros = diabdrugpatzeros + temp31
                
    diabdrugpatzeros = np.where(diabdrugpatzeros > 0.5, 1, 0)
    diabdrugpat.append(allergy[k][allergy[k].allergyname == np.multiply(diabdrugpatzeros, allergy[k].allergyname)])
    searchlist.clear()

    allergy_m.extend(diabdrugpat)
    allergy_m[k] = allergy_m[k].sort_values(by = ['allergyoffset'])
    
    if len(allergy_m[k]) != 0:
        y3 = y3 + 1
        print('The Corresponding Index is', k, 'with', len(allergy_m[k]), 'sample(s)')
        
print('allergy_m has', y3, 'patients')
        
### Table 4: apacheApsVar ####################################################################################################

## eyes + motor + verbal -----> meds ##########################################
apacheapsvar_m = copy.deepcopy(apacheapsvar)
meds = []

for k in range(len(apacheapsvar)):
    apacheapsvar_m[k].eyes.replace(-1, 1, inplace=True)
    apacheapsvar_m[k].motor.replace(-1, 1, inplace=True)
    apacheapsvar_m[k].verbal.replace(-1, 1, inplace=True)
    for x4 in range(len(apacheapsvar[k])):
        meds.append((apacheapsvar_m[k].iloc[x4].eyes + apacheapsvar_m[k].iloc[x4].motor + apacheapsvar_m[k].iloc[x4].verbal)/15)
        
    apacheapsvar_m[k] = apacheapsvar_m[k].drop(apacheapsvar_m[k].columns[[6, 7, 8, 9]], axis=1)
    apacheapsvar_m[k].insert(6, 'meds', meds, True)
    meds.clear()

# Table 5: apachePatientResult ###############################################################################################

#129 Expired ---> sample: 343 #################################################
apachepatientresult_m = copy.deepcopy(apachepatientresult)
y5 = 0

for k in range(len(apachepatientresult)):
    apachepatientresult_m[k] = apachepatientresult_m[k].sort_values(by = ['actualiculos'], ascending = False)
    apachepatientresult_m[k] = apachepatientresult_m[k].drop(apachepatientresult_m[k].columns[[16, 17, 18, 19, 20, 21, 22, 23]], axis=1)
    
    if len(apachepatientresult_m[k]) != 0:
        if (apachepatientresult_m[k].actualhospitalmortality.iloc[0] == 'EXPIRED'):
            y5 = y5 + 1
            print('The Corresponding Index is', k)
        
print('apachepatientresult_m has', y5, 'expired patients')

# Table 6: apachePredVar ##################################################################################################### 

#27 diabetic expired (out of 369 diabetic) ---> sample: 1458 ##################
apachepredvar_m = copy.deepcopy(apachepredvar)
y6 = 0
y60 = 0
diabetic = []
diabeticdeads = []

for k in range(len(apachepredvar)):
    apachepredvar_m[k] = apachepredvar_m[k].sort_values(by = ['age', 'visitnumber'], ascending = [True, True])
    apachepredvar_m[k].eyes.replace(-1, 1, inplace=True)
    apachepredvar_m[k].motor.replace(-1, 1, inplace=True)
    apachepredvar_m[k].verbal.replace(-1, 1, inplace=True)
    for x6 in range(len(apachepredvar[k])):
        meds.append((apachepredvar_m[k].iloc[x6].eyes + apachepredvar_m[k].iloc[x6].motor + apachepredvar_m[k].iloc[x6].verbal)/15)
    
    apachepredvar_m[k] = apachepredvar_m[k].drop(apachepredvar_m[k].columns[[13, 14, 15, 16]], axis=1)
    apachepredvar_m[k].insert(12, 'meds', meds ,True)
    meds.clear()

    if len(apachepredvar_m[k]) != 0:
        if (apachepredvar_m[k].diabetes.iloc[0] == 1):
            diabetic.append(k)
            y6 = y6 + 1
            if (apachepredvar_m[k].diedinhospital.sum() != 0):
                y60 = y60 + 1
                diabeticdeads.append(apachepredvar_m[k])
                print('The Corresponding Index is', k)
        
print('apachepredvar_m has', y60, 'expired diabetic patients among', y6, 'diabetic patients')

### Table 13: diagnosis ######################################################################################################

#291 patients ---> sample: 1344, 298 & 317 (without True!) ######################
druglist3 = ['diABeT', 'hyPoGlYcemIA', 'SUlfOnYlurEA', 'HypErGlYCemIa']
diagnosis_m = []
y13 = 0

for k in range(len(diagnosis)):
    diabdrugpat = []
    for x13 in range(len(druglist3)):
        searchlist.append([druglist3[x13].lower(), druglist3[x13].upper(), druglist3[x13].capitalize()])
        diabdrugpat.append(diagnosis[k][(diagnosis[k].diagnosisstring.str.count(searchlist[x13][0]) + diagnosis[k].diagnosisstring.str.count(searchlist[x13][1]) + diagnosis[k].diagnosisstring.str.count(searchlist[x13][2]) == 1)])
    
    diabdrugpat = pd.concat(diabdrugpat[:], axis=0, join='outer', ignore_index=False)
    diabdrugpat = diabdrugpat.sort_values(by = ['diagnosisoffset'])
    diagnosis_m.append(diabdrugpat)
    searchlist.clear()

    if len(diagnosis_m[k]) != 0:
        y13 = y13 + 1
        print('The Corresponding Index is', k, 'with', len(diagnosis_m[k]), 'sample(s)')
        
print('diagnosis_m has', y13, 'patients')

### Table 15: infusionDrug ###################################################################################################

#117 patients ---> sample: 1664 (insulin) & 1768 (lactate) ##############
infusiondrug_m = []
y15 = 0

for k in range(len(infusiondrug)):
    diabdrugpat = []
    for x15 in range(len(druglist1)):
        searchlist.append([druglist1[x15].lower(), druglist1[x15].upper(), druglist1[x15].capitalize()])
        diabdrugpat.append(infusiondrug[k][(infusiondrug[k].drugname.str.count(searchlist[x15][0]) + infusiondrug[k].drugname.str.count(searchlist[x15][1]) + infusiondrug[k].drugname.str.count(searchlist[x15][2]) == 1)])
    
    diabdrugpat = pd.concat(diabdrugpat[:], axis=0, join='outer', ignore_index=False)
    diabdrugpat = diabdrugpat.sort_values(by = ['infusionoffset'])
    infusiondrug_m.append(diabdrugpat)
    searchlist.clear()

    if len(infusiondrug_m[k]) != 0:
        y15 = y15 + 1
        print('The Corresponding Index is', k, 'with', len(infusiondrug_m[k]), 'sample(s)')
        
print('infusiondrug_m has', y15, 'patients')

### Table 16: intakeOutput ###################################################################################################

#69 patients ---> sample: 1 (lactate), 773 (insulin) ########################
intakeoutput_m = []
y16 = 0

for k in range(len(intakeoutput)):
    diabdrugpat = []
    for x16 in range(len(druglist1)):
        searchlist.append([druglist1[x16].lower(), druglist1[x16].upper(), druglist1[x16].capitalize()])
        diabdrugpat.append(intakeoutput[k][(intakeoutput[k].celllabel.str.count(searchlist[x16][0]) + intakeoutput[k].celllabel.str.count(searchlist[x16][1]) + intakeoutput[k].celllabel.str.count(searchlist[x16][2]) == 1)])
    
    diabdrugpat = pd.concat(diabdrugpat[:], axis=0, join='outer', ignore_index=False)
    diabdrugpat = diabdrugpat.sort_values(by = ['intakeoutputoffset'])
    intakeoutput_m.append(diabdrugpat)
    searchlist.clear()

    if len(intakeoutput_m[k]) != 0:
        y16 = y16 + 1
        print('The Corresponding Index is', k, 'with', len(intakeoutput_m[k]), 'sample(s)')
        
print('intakeoutput_m has', y16, 'patients')

### Table 17: lab ############################################################################################################

#1803 patients ---> sample: 1748 ##############################################
# lactate: 811
# bedside glucose: 1245
# glucose - CSF: 19
# glucose: 857 (2121-19-1245)
# mmol/l = mg/dl * 18.0182
lab_m = []
y17 = 0

for k in range(len(lab)):
    diabdrugpat = []
    for x17 in range(len(druglist1)):
        searchlist.append([druglist1[x17].lower(), druglist1[x17].upper(), druglist1[x17].capitalize()])
        diabdrugpat.append(lab[k][(lab[k].labname.str.count(searchlist[x17][0]) + lab[k].labname.str.count(searchlist[x17][1]) + lab[k].labname.str.count(searchlist[x17][2]) == 1)])
    
    diabdrugpat = pd.concat(diabdrugpat[:], axis=0, join='outer', ignore_index=False)
    diabdrugpat = diabdrugpat.sort_values(by = ['labresultoffset'])
    lab_m.append(diabdrugpat)
    searchlist.clear()

    if len(lab_m[k]) != 0:
        y17 = y17 + 1
        print('The Corresponding Index is', k, 'with', len(lab_m[k]), 'sample(s)')
        
print('lab_m has', y17, 'patients')

### Table 18: medication #####################################################################################################

#eICU[17][(eICU[17].drughiclseqno == 807)]
#eICU[17][(eICU[17].drughiclseqno == 19078)]
#eICU[17][(eICU[17].drugname.str.count('GLUCA') == 1)]

#826 patients ---> smaple: 514, 1 ############################################

medication_m = []
y18 = 0

for k in range(len(medication)):
    diabdrugpat = []
    diabdrugpatzeros = np.zeros(len(medication[k]))
    for x18 in range(len(druglist1)):
        searchlist.append([druglist1[x18].lower(), druglist1[x18].upper(), druglist1[x18].capitalize()])
        temp18 = medication[k].drugname.str.count(searchlist[x18][0]).fillna(0) + medication[k].drugname.str.count(searchlist[x18][1]).fillna(0) + medication[k].drugname.str.count(searchlist[x18][2]).fillna(0)
        diabdrugpatzeros = diabdrugpatzeros + temp18
        for x19 in range(len(drughicllist)):
            temp19 = (medication[k].drughiclseqno == drughicllist[x19])*1
            diabdrugpatzeros = diabdrugpatzeros + temp19
                
    diabdrugpatzeros = np.where(diabdrugpatzeros > 0.5, 1, 0)
    diabdrugpat.append(medication[k][medication[k].patientunitstayid == np.multiply(diabdrugpatzeros, medication[k].patientunitstayid)])
    searchlist.clear()

    medication_m.extend(diabdrugpat)
    medication_m[k] = medication_m[k].sort_values(by = ['drugstartoffset', 'drugorderoffset'], ascending = [True, True])
    
    if len(medication_m[k]) != 0:
        y18 = y18 + 1
        print('The Corresponding Index is', k, 'with', len(medication_m[k]), 'sample(s)')
        
print('medication_m has', y18, 'patients')

### Table 23: nurseCharting ##################################################################################################
#481 patients ---> sample: 206 ################################################
druglist4 = ['gLuCoSE']
nursechart_m = []
y23 = 0

for k in range(len(nursechart)):
    diabdrugpat = []
    for x23 in range(len(druglist4)):
        searchlist.append([druglist4[x23].lower(), druglist4[x23].upper(), druglist4[x23].capitalize()])
        diabdrugpat.append(nursechart[k][(nursechart[k].nursingchartcelltypevallabel.str.count(searchlist[x23][0]) + nursechart[k].nursingchartcelltypevallabel.str.count(searchlist[x23][1]) + nursechart[k].nursingchartcelltypevallabel.str.count(searchlist[x23][2]) == 1)])
    
    diabdrugpat = pd.concat(diabdrugpat[:], axis=0, join='outer', ignore_index=False)
    diabdrugpat = diabdrugpat.sort_values(by = ['nursingchartoffset'])
    nursechart_m.append(diabdrugpat)
    searchlist.clear()

    if len(nursechart_m[k]) != 0:
        y23 = y23 + 1
        print('The Corresponding Index is', k, 'with', len(nursechart_m[k]), 'sample(s)')
        
print('nursechart_m has', y23, 'patients')

#a = []
#for k in range(len(nursechart_m)):
#    a.append(len(nursechart_m[k]))
#    
#a.index(max(a))    

### Table 24: pastHistory ####################################################################################################

#509 patients ---> smaple: 1176, WEIRD samples: 152 (34) - 232 (3) - 339 (6)
druglist5 = ['Non-Insulin Dependent Diabetes/medication dependent', 
             'Non-Insulin Dependent Diabetes/non-medication dependent', 
             'Insulin Dependent Diabetes/insulin dependent diabetes']
pasthistory_m = []
y24 = 0

y241 = 0
y242 = 0
y243 = 0

y24123 = 0

for k in range(len(pasthistory)):
    diabdrugpat = []
    for x24 in range(len(druglist5)):
        diabdrugpat.append(pasthistory[k][(pasthistory[k].pasthistorypath.str.count(druglist5[x24]) == 1)])
    
    # if (len(diabdrugpat[0]) * len(diabdrugpat[1]) * len(diabdrugpat[2])) != 0:
    #     y24123 = y24123 + 1                                                     
    #     print('The Corresponding Index for 24123 is', k)
    # elif (len(diabdrugpat[0]) * len(diabdrugpat[2])) != 0:
    #     y242 = y242 + 1                                                   #34
    #     print('The Corresponding Index for 242 is', k)
    # elif (len(diabdrugpat[1]) * len(diabdrugpat[2])) != 0:
    #     y243 = y243 + 1                                                    #3
    #     print('The Corresponding Index for 243 is', k)
    # elif (len(diabdrugpat[0]) * len(diabdrugpat[1])) != 0:
    #     y241 = y241 + 1                                                    #6
    #     print('The Corresponding Index for 241 is', k)
    if len(diabdrugpat[0]) != 0:
        y241 = y241 + 1
    elif len(diabdrugpat[1]) != 0:
        y242 = y242 + 1
    elif len(diabdrugpat[2]) != 0:
        y243 = y243 + 1

    diabdrugpat = pd.concat(diabdrugpat[:], axis=0, join='outer', ignore_index=False)
    diabdrugpat = diabdrugpat.sort_values(by = ['pasthistoryoffset'])
    pasthistory_m.append(diabdrugpat)

    if len(pasthistory_m[k]) != 0:
        y24 = y24 + 1
        print('The Corresponding Index is', k, 'with', len(pasthistory_m[k]), 'sample(s)')
        
print('pasthistory_m has', y24, 'patients')
print('Non-Insulin Dependent Diabetes/medication dependent are:', y241, 'patients')         #239
print('Non-Insulin Dependent Diabetes/non-medication dependent are:', y242, 'patients')     #54
print('Insulin Dependent Diabetes/insulin dependent are:', y243, 'patients')                #216

# y246 = 0
# for k in range(len(pasthistory_m)):
#     if len(apachepredvar_m[k]) != 0:
#         if len(pasthistory_m[k]) * apachepredvar_m[k].diabetes.iloc[0] != 0:
#             y246 = y246 + 1
#             print('The Corresponding Index is', k)
# print('intersection has', y246, 'patients')                      #372 patients

### Table 25: patient ########################################################################################################
#117 patients ---> sample: 179, 331, 1321
patient_m = []
y25 = 0

for k in range(len(patient)):
    diabdrugpat = []
    diabdrugpatzeros = np.zeros(len(patient[k]))
    for x25 in range(len(druglist1)):
        searchlist.append([druglist1[x25].lower(), druglist1[x25].upper(), druglist1[x25].capitalize()])
        temp25 = patient[k].apacheadmissiondx.str.count(searchlist[x25][0]).fillna(0) + patient[k].apacheadmissiondx.str.count(searchlist[x25][1]).fillna(0) + patient[k].apacheadmissiondx.str.count(searchlist[x25][2]).fillna(0)
        diabdrugpatzeros = diabdrugpatzeros + temp25
                
    diabdrugpatzeros = np.where(diabdrugpatzeros > 0.5, 1, 0)
    diabdrugpat.append(patient[k][patient[k].apacheadmissiondx == np.multiply(diabdrugpatzeros, patient[k].apacheadmissiondx)])
    searchlist.clear()

    patient_m.extend(diabdrugpat)
    patient_m[k] = patient_m[k].sort_values(by = ['hospitaldischargeyear', 'hospitaladmitoffset'], ascending = [True, True])
    
    if len(patient_m[k]) != 0:
        y25 = y25 + 1
        print('The Corresponding Index is', k, 'with', len(patient_m[k]), 'sample(s)')
        
print('patient_m has', y25, 'patients')

y325 = 0
for k in range(len(admissiondx_m)):
    if (len(admissiondx_m[k]) * len(patient_m[k])) != 0:
        y325 = y325 + 1
print('intersection has', y325, 'patients')                       #117 patients

# i = 0
# for k in range(len(patient_m)):
#     if len(patient_m[k]) != 0:
#         if patient_m[k].hospitaldischargestatus.iloc[0] == 'Expired':
#             i = i + 1

### Table 29: treatment ######################################################################################################

#333 patients ---> sample: 762 ################################################
treatment_m = []
y29 = 0

for k in range(len(treatment)):
    diabdrugpat = []
    for x29 in range(len(druglist1)):
        searchlist.append([druglist1[x29].lower(), druglist1[x29].upper(), druglist1[x29].capitalize()])
        diabdrugpat.append(treatment[k][(treatment[k].treatmentstring.str.count(searchlist[x29][0]) + treatment[k].treatmentstring.str.count(searchlist[x29][1]) + treatment[k].treatmentstring.str.count(searchlist[x29][2]) == 1)])
    
    diabdrugpat = pd.concat(diabdrugpat[:], axis=0, join='outer', ignore_index=False)
    diabdrugpat = diabdrugpat.sort_values(by = ['treatmentoffset'])
    treatment_m.append(diabdrugpat)
    searchlist.clear()

    if len(treatment_m[k]) != 0:
        y29 = y29 + 1
        print('The Corresponding Index is', k, 'with', len(treatment_m[k]), 'sample(s)')
        
print('treatment_m has', y29, 'patients')

# a = []
# for k in range(len(treatment_m)):
#     a.append(len(treatment_m[k]))
    
# a.index(max(a))
##############################################################################################################################

y56 = 0
for k in range(len(apachepredvar_m)):
    if len(apachepredvar_m[k]) * len(apachepatientresult_m[k]) != 0:
        if ((apachepredvar_m[k].diedinhospital.sum() != 0)*1) * ((apachepatientresult_m[k].actualhospitalmortality.iloc[0] == 'EXPIRED')*1) == 1:
            y56 = y56 + 1
            print('The corresponding index is:', k)
print('There are', y56, 'intersections between apachepatientresult and apachepredvar')


y29 = 0
for k in range(len(treatment_m)):
    if len(treatment_m[k]) != 0:
        if sum(treatment_m[k].activeupondischarge) == 0:
            print('Patient', k, 'has a hospital discharge status of ', patient[k].hospitaldischargestatus.iloc[0])
            
expiredapachepr = []
for k in range(len(apachepatientresult_m)):
    if len(apachepatientresult_m[k]) != 0:
        if (apachepatientresult_m[k].actualhospitalmortality.iloc[0] == 'EXPIRED'):
            expiredapachepr.append(k)
        
expiredapachepv = []
for k in range(len(apachepredvar_m)):
    if len(apachepredvar_m[k]) != 0:
        if (apachepredvar_m[k].diedinhospital.sum()) != 0:
            expiredapachepv.append(k)
            
set(expiredapachepr).issubset(expiredapachepv)

# y560 = 0
# for k in range(len(diabeticdeads)):
#     if diabeticdeads[k] in expiredapachepr:
#         y560 = y560 + 1
        
# y560


import re
Match = []

for k in range(len(medication_m)):
    if len(medication_m[k]) != 0:
        match = []
        match.insert(0, k)
        for k1 in range(len(medication_m[k])):
            if pd.isnull(medication_m[k].iloc[k1].dosage) == False:
                match.append(re.findall(r'[\d.,]+', medication_m[k].iloc[k1].dosage.replace(',', '')))
                
        Match.append(match)
        
##############################################################################################################################

# Separating Glucose in lab_m #################################################
#1802 lab_mg
lab_mg = []
y171 = 0

for k in range(len(lab)):
    diabdrugpat = []
    for x171 in range(len(druglist4)):
        searchlist.append([druglist4[x171].lower(), druglist4[x171].upper(), druglist4[x171].capitalize()])
        diabdrugpat.append(lab[k][(lab[k].labname.str.count(searchlist[x171][0]) + lab[k].labname.str.count(searchlist[x171][1]) + lab[k].labname.str.count(searchlist[x171][2]) == 1)])
    
    diabdrugpat = pd.concat(diabdrugpat[:], axis=0, join='outer', ignore_index=False)
    diabdrugpat = diabdrugpat.sort_values(by = ['labresultoffset'])
    lab_mg.append(diabdrugpat)
    searchlist.clear()

    if len(lab_mg[k]) != 0:
        y171 = y171 + 1
        print('The Corresponding Index is', k, 'with', len(lab_mg[k]), 'sample(s)')
        
print('lab_mg has', y171, 'patients')

# Separating Lactate in lab_m #################################################
#722 lab_ml, lactate and no glucose: 185
druglist6 = ['lACtAte']
lab_ml = []
y172 = 0

for k in range(len(lab)):
    diabdrugpat = []
    for x172 in range(len(druglist6)):
        searchlist.append([druglist6[x172].lower(), druglist6[x172].upper(), druglist6[x172].capitalize()])
        diabdrugpat.append(lab[k][(lab[k].labname.str.count(searchlist[x172][0]) + lab[k].labname.str.count(searchlist[x172][1]) + lab[k].labname.str.count(searchlist[x172][2]) == 1)])
    
    diabdrugpat = pd.concat(diabdrugpat[:], axis=0, join='outer', ignore_index=False)
    diabdrugpat = diabdrugpat.sort_values(by = ['labresultoffset'])
    lab_ml.append(diabdrugpat)
    searchlist.clear()
    
    if len(lab_ml[k]) != 0:
        y172 = y172 + 1
        print('The Corresponding Index is', k, 'with', len(lab_ml[k]), 'sample(s)')
        
print('lab_ml has', y172, 'patients')

##############################################################################################################################
patient_weight = copy.deepcopy(patient)
for k in range(len(patient_weight)):
    patient_weight[k].admissionweight.fillna(0, inplace=True)
    patient_weight[k].dischargeweight.fillna(0, inplace=True)
    for k2 in range(len(patient_weight[k]))[::-1]:
        if patient_weight[k].admissionweight.iloc[k2] == 0:
            if patient_weight[k].dischargeweight.iloc[k2] != 0:
                patient_weight[k].admissionweight.iloc[k2] = patient_weight[k].dischargeweight.iloc[k2]
        if patient_weight[k].admissionweight.iloc[k2] != 0:
            if patient_weight[k].dischargeweight.iloc[k2] == 0:
                patient_weight[k].dischargeweight.iloc[k2] = patient_weight[k].admissionweight.iloc[k2]
        if patient_weight[k].admissionweight.iloc[k2] == 0:
            if patient_weight[k].dischargeweight.iloc[k2] == 0:
                patient_weight[k] = patient_weight[k].drop([k2], axis=0)
                    
lab_mlc = copy.deepcopy(lab_ml)
ML_primary = []

for k in range(len(lab_mlc)):
    if len(lab_mlc[k]) != 0:
        for k2 in range(len(lab_mlc[k])):
            if np.isnan(lab_mlc[k].labresult.iloc[k2]):
                lab_mlc[k].labresult.iloc[k2] = re.findall(r'\d.+', str(lab_mlc[k].labresulttext.iloc[k2]))[0]
                lab_mlc[k].labresult.iloc[k2] = float(lab_mlc[k].labresult.iloc[k2])
        if len(lab_mg[k]) != 0:
            ML_init = pd.DataFrame(columns = ['uniquepid', 'age', 'gender', 'admissionweight', 'dischargeweight', 'minimumglucose', 'meanglucose', 'maximumglucose', 'minimumlactate', 'meanlactate', 'maximumlactate', 'diabetes', 'mortality'], index=[k])
            if len(apachepredvar_m[k]) == 0:
                ML_init.loc[k] = (lab_mlc[k].uniquepid.iloc[0], patient[k].age.iloc[0], patient[k].gender.iloc[0], patient_weight[k].admissionweight.mean(), patient_weight[k].dischargeweight.mean(), lab_mg[k].labresult.min(), lab_mg[k].labresult.mean(), lab_mg[k].labresult.max(), (lab_mlc[k].labresult.min()*18.0182), (lab_mlc[k].labresult.mean()*18.0182), (lab_mlc[k].labresult.max()*18.0182), 0, patient[k].hospitaldischargestatus.iloc[0])
            else: 
                ML_init.loc[k] = (lab_mlc[k].uniquepid.iloc[0], patient[k].age.iloc[0], patient[k].gender.iloc[0], patient_weight[k].admissionweight.mean(), patient_weight[k].dischargeweight.mean(), lab_mg[k].labresult.min(), lab_mg[k].labresult.mean(), lab_mg[k].labresult.max(), (lab_mlc[k].labresult.min()*18.0182), (lab_mlc[k].labresult.mean()*18.0182), (lab_mlc[k].labresult.max()*18.0182), apachepredvar_m[k].diabetes.iloc[0], patient[k].hospitaldischargestatus.iloc[0])
            ML_primary.append(ML_init)
            
ML_primary = pd.concat(ML_primary[:], axis=0, join='outer', ignore_index=False)
#ML_primary = ML_primary.rename(columns={'Unnamed: 0': 'patientnumber'})
ML_primary.gender = np.where(ML_primary.gender == 'Female', 1, 0)
ML_primary.age = np.where(ML_primary.age == '> 89', 90, ML_primary.age)
# ML_primary['uniquepid'] = ML_primary['uniquepid'].astype(int)
ML_primary['age'] = ML_primary['age'].astype(int)
ML_primary['gender'] = ML_primary['gender'].astype(int)
ML_primary['minimumglucose'] = ML_primary['minimumglucose'].astype(float)
ML_primary['meanglucose'] = ML_primary['meanglucose'].astype(float)
ML_primary['maximumglucose'] = ML_primary['maximumglucose'].astype(float)
ML_primary['minimumlactate'] = ML_primary['minimumlactate'].astype(float)
ML_primary['meanlactate'] = ML_primary['meanlactate'].astype(float)
ML_primary['maximumlactate'] = ML_primary['maximumlactate'].astype(float)
ML_primary['diabetes'] = ML_primary['diabetes'].astype(int)   
for k in ML_primary.index:
    if pd.isna(ML_primary.loc[k].admissionweight):
        # print(k)
        ML_primary.admissionweight.loc[k] = infusiondrug[k].patientweight.mean()
        ML_primary.dischargeweight.loc[k] = infusiondrug[k].patientweight.mean()




ML_primary['admissionweight'] = ML_primary['admissionweight'].astype(float)
ML_primary['dischargeweight'] = ML_primary['dischargeweight'].astype(float)

for k in ML_primary.index:
    if pd.isna(ML_primary.loc[k].mortality):
        # print(k)
        ML_primary.mortality.loc[k] = apachepredvar[k].diedinhospital.sum()
        ML_primary.mortality.loc[k] = np.where(ML_primary.mortality.loc[k] == 0, 'Alive', 'Expired')

ML_primary['mortality'] = ML_primary['mortality'].astype(str)

ML_secondary = copy.deepcopy(ML_primary)
for k in ML_secondary.index:
    if ML_secondary.loc[k].isna().any():
        ML_secondary.drop([k], inplace=True)
        
        
ML_secondary.to_csv('ML_ready.csv')
##############################################################################################################################
