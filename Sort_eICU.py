import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#from scipy import interpolate
#import pylab as py
#import matplotlib.pyplot as plt
#import pdvega

import os
import glob

filepath = os.chdir('C:/Users/Kaveh Zaferanloui/Desktop/Norway/Thesis/Data/eICU/eICU - Full Version/eicu-collaborative-research-database-2.0/CSVs')
##a = os.listdir(filepath)
eICU = glob.glob('*.{}'.format('csv'))

# read csv files
for i in range(len(eICU)):
    eICU[i] = pd.read_csv(eICU[i])

### Main tables
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

### Table 25: patient sorted based on patienthealthsystemid
patient_num = eICU[24].patienthealthsystemstayid.unique()
patient = []
for j in range(len(patient_num)):
    patient.append(eICU[24][eICU[24].patienthealthsystemstayid == patient_num[j]].sort_values(by = ['unitvisitnumber']))
    patient[j].reset_index(drop=True, inplace=True)    

for k in range(len(patient_num)):
### Table 1: admissiondrug
    admissiondrugzeros = np.zeros(len(eICU[0].patientunitstayid))
    for l0 in range(len(patient[k])):
        admissiondrugzeros = admissiondrugzeros + ((eICU[0].patientunitstayid == patient[k].patientunitstayid[l0])*1)

    admissiondrug.append(eICU[0][eICU[0].patientunitstayid == np.multiply(admissiondrugzeros, eICU[0].patientunitstayid)])
    
### Table 2: admissiondx
    admissiondxzeros = np.zeros(len(eICU[1].patientunitstayid))
    for l1 in range(len(patient[k])):
        admissiondxzeros = admissiondxzeros + ((eICU[1].patientunitstayid == patient[k].patientunitstayid[l1])*1)

    admissiondx.append(eICU[1][eICU[1].patientunitstayid == np.multiply(admissiondxzeros, eICU[1].patientunitstayid)])
    
### Table 3: allergy
    allergyzeros = np.zeros(len(eICU[2].patientunitstayid))
    for l2 in range(len(patient[k])):
        allergyzeros = allergyzeros + ((eICU[2].patientunitstayid == patient[k].patientunitstayid[l2])*1)

    allergy.append(eICU[2][eICU[2].patientunitstayid == np.multiply(allergyzeros, eICU[2].patientunitstayid)])
    
### Table 4: ApacheApsVar
    apacheavzeros = np.zeros(len(eICU[3].patientunitstayid))
    for l3 in range(len(patient[k])):
        apacheavzeros = apacheavzeros + ((eICU[3].patientunitstayid == patient[k].patientunitstayid[l3])*1)

    apacheapsvar.append(eICU[3][eICU[3].patientunitstayid == np.multiply(apacheavzeros, eICU[3].patientunitstayid)])

### Table 5: ApachePatientResult
    apacheprzeros = np.zeros(len(eICU[4].patientunitstayid))
    for l4 in range(len(patient[k])):
        apacheprzeros = apacheprzeros + ((eICU[4].patientunitstayid == patient[k].patientunitstayid[l4])*1)

    apachepatientresult.append(eICU[4][eICU[4].patientunitstayid == np.multiply(apacheprzeros, eICU[4].patientunitstayid)])

### Table 6: ApachePredVar
    apachepvzeros = np.zeros(len(eICU[5].patientunitstayid))
    for l5 in range(len(patient[k])):
        apachepvzeros = apachepvzeros + ((eICU[5].patientunitstayid == patient[k].patientunitstayid[l5])*1)

    apachepredvar.append(eICU[5][eICU[5].patientunitstayid == np.multiply(apachepvzeros, eICU[5].patientunitstayid)])

### Table 7: carePlanCareProvider
    cpcpzeros = np.zeros(len(eICU[6].patientunitstayid))
    for l6 in range(len(patient[k])):
        cpcpzeros = cpcpzeros + ((eICU[6].patientunitstayid == patient[k].patientunitstayid[l6])*1)

    cpcareprovider.append(eICU[6][eICU[6].patientunitstayid == np.multiply(cpcpzeros, eICU[6].patientunitstayid)])

### Table 8: carePlanEOL
    cpeolzeros = np.zeros(len(eICU[7].patientunitstayid))
    for l7 in range(len(patient[k])):
        cpeolzeros = cpeolzeros + ((eICU[7].patientunitstayid == patient[k].patientunitstayid[l7])*1)

    cpeol.append(eICU[7][eICU[7].patientunitstayid == np.multiply(cpeolzeros, eICU[7].patientunitstayid)])

### Table 9: carePlanGeneral
    cpgenzeros = np.zeros(len(eICU[8].patientunitstayid))
    for l8 in range(len(patient[k])):
        cpgenzeros = cpgenzeros + ((eICU[8].patientunitstayid == patient[k].patientunitstayid[l8])*1)

    cpgeneral.append(eICU[8][eICU[8].patientunitstayid == np.multiply(cpgenzeros, eICU[8].patientunitstayid)])

### Table 10: carePlanGoal
    cpgoalzeros = np.zeros(len(eICU[9].patientunitstayid))
    for l9 in range(len(patient[k])):
        cpgoalzeros = cpgoalzeros + ((eICU[9].patientunitstayid == patient[k].patientunitstayid[l9])*1)

    cpgoal.append(eICU[9][eICU[9].patientunitstayid == np.multiply(cpgoalzeros, eICU[9].patientunitstayid)])

### Table 11: carePlanInfectiousDisease
    cpinfdiseasezeros = np.zeros(len(eICU[10].patientunitstayid))
    for l10 in range(len(patient[k])):
        cpinfdiseasezeros = cpinfdiseasezeros + ((eICU[10].patientunitstayid == patient[k].patientunitstayid[l10])*1)

    cpinfectiousdisease.append(eICU[10][eICU[10].patientunitstayid == np.multiply(cpinfdiseasezeros, eICU[10].patientunitstayid)])

### Table 12: customLab
    customlabzeros = np.zeros(len(eICU[11].patientunitstayid))
    for l11 in range(len(patient[k])):
        customlabzeros = customlabzeros + ((eICU[11].patientunitstayid == patient[k].patientunitstayid[l11])*1)

    customlab.append(eICU[11][eICU[11].patientunitstayid == np.multiply(customlabzeros, eICU[11].patientunitstayid)])

### Table 13: diagnosis
    diagnosiszeros = np.zeros(len(eICU[12].patientunitstayid))
    for l12 in range(len(patient[k])):
        diagnosiszeros = diagnosiszeros + ((eICU[12].patientunitstayid == patient[k].patientunitstayid[l12])*1)

    diagnosis.append(eICU[12][eICU[12].patientunitstayid == np.multiply(diagnosiszeros, eICU[12].patientunitstayid)])

### Table 14: hospital
    hospitalzeros = np.zeros(len(eICU[13].hospitalid))
    #for l13 in range(len(patient[k])):
    hospitalzeros = hospitalzeros + ((eICU[13].hospitalid == patient[k].hospitalid[0])*1)

    hospital.append(eICU[13][eICU[13].hospitalid == np.multiply(hospitalzeros, eICU[13].hospitalid)])
    hospital[k].insert(0, 'patienthealthsystemstayid', [patient[k].patienthealthsystemstayid[0]], True)

### Table 15: infusionDrug
    infusiondrugzeros = np.zeros(len(eICU[14].patientunitstayid))
    for l14 in range(len(patient[k])):
        infusiondrugzeros = infusiondrugzeros + ((eICU[14].patientunitstayid == patient[k].patientunitstayid[l14])*1)

    infusiondrug.append(eICU[14][eICU[14].patientunitstayid == np.multiply(infusiondrugzeros, eICU[14].patientunitstayid)])

### Table 16: intakeOutput
    intakeoutputzeros = np.zeros(len(eICU[15].patientunitstayid))
    for l15 in range(len(patient[k])):
        intakeoutputzeros = intakeoutputzeros + ((eICU[15].patientunitstayid == patient[k].patientunitstayid[l15])*1)

    intakeoutput.append(eICU[15][eICU[15].patientunitstayid == np.multiply(intakeoutputzeros, eICU[15].patientunitstayid)])

### Table 17: lab
    labzeros = np.zeros(len(eICU[16].patientunitstayid))
    for l16 in range(len(patient[k])):
        labzeros = labzeros + ((eICU[16].patientunitstayid == patient[k].patientunitstayid[l16])*1)

    lab.append(eICU[16][eICU[16].patientunitstayid == np.multiply(labzeros, eICU[16].patientunitstayid)])

### Table 18: medication
    medicationzeros = np.zeros(len(eICU[17].patientunitstayid))
    for l17 in range(len(patient[k])):
        medicationzeros = medicationzeros + ((eICU[17].patientunitstayid == patient[k].patientunitstayid[l17])*1)

    medication.append(eICU[17][eICU[17].patientunitstayid == np.multiply(medicationzeros, eICU[17].patientunitstayid)])
## len(eICU[17].drugname[(eICU[17].drugname.str.count('metformin') != 0)].dropna())

### Table 19: microLab
    microlabzeros = np.zeros(len(eICU[18].patientunitstayid))
    for l18 in range(len(patient[k])):
        microlabzeros = microlabzeros + ((eICU[18].patientunitstayid == patient[k].patientunitstayid[l18])*1)

    microlab.append(eICU[18][eICU[18].patientunitstayid == np.multiply(microlabzeros, eICU[18].patientunitstayid)])

### Table 20: note
    notezeros = np.zeros(len(eICU[19].patientunitstayid))
    for l19 in range(len(patient[k])):
        notezeros = notezeros + ((eICU[19].patientunitstayid == patient[k].patientunitstayid[l19])*1)

    note.append(eICU[19][eICU[19].patientunitstayid == np.multiply(notezeros, eICU[19].patientunitstayid)])

### Table 21: nurseAssessment
    nurseassesszeros = np.zeros(len(eICU[20].patientunitstayid))
    for l20 in range(len(patient[k])):
        nurseassesszeros = nurseassesszeros + ((eICU[20].patientunitstayid == patient[k].patientunitstayid[l20])*1)

    nurseassess.append(eICU[20][eICU[20].patientunitstayid == np.multiply(nurseassesszeros, eICU[20].patientunitstayid)])

### Table 22: nurseCare
    nursecarezeros = np.zeros(len(eICU[21].patientunitstayid))
    for l21 in range(len(patient[k])):
        nursecarezeros = nursecarezeros + ((eICU[21].patientunitstayid == patient[k].patientunitstayid[l21])*1)

    nursecare.append(eICU[21][eICU[21].patientunitstayid == np.multiply(nursecarezeros, eICU[21].patientunitstayid)])

### Table 23: nurseCharting
    nursechartzeros = np.zeros(len(eICU[22].patientunitstayid))
    for l22 in range(len(patient[k])):
        nursechartzeros = nursechartzeros + ((eICU[22].patientunitstayid == patient[k].patientunitstayid[l22])*1)

    nursechart.append(eICU[22][eICU[22].patientunitstayid == np.multiply(nursechartzeros, eICU[22].patientunitstayid)])

### Table 24: pastHistory
    pasthistoryzeros = np.zeros(len(eICU[23].patientunitstayid))
    for l23 in range(len(patient[k])):
        pasthistoryzeros = pasthistoryzeros + ((eICU[23].patientunitstayid == patient[k].patientunitstayid[l23])*1)

    pasthistory.append(eICU[23][eICU[23].patientunitstayid == np.multiply(pasthistoryzeros, eICU[23].patientunitstayid)])

### Table 26: physicalExam
    physicalexamzeros = np.zeros(len(eICU[25].patientunitstayid))
    for l25 in range(len(patient[k])):
        physicalexamzeros = physicalexamzeros + ((eICU[25].patientunitstayid == patient[k].patientunitstayid[l25])*1)

    physicalexam.append(eICU[25][eICU[25].patientunitstayid == np.multiply(physicalexamzeros, eICU[25].patientunitstayid)])

### Table 27: respiratoryCare
    respiratorycarezeros = np.zeros(len(eICU[26].patientunitstayid))
    for l26 in range(len(patient[k])):
        respiratorycarezeros = respiratorycarezeros + ((eICU[26].patientunitstayid == patient[k].patientunitstayid[l26])*1)

    respiratorycare.append(eICU[26][eICU[26].patientunitstayid == np.multiply(respiratorycarezeros, eICU[26].patientunitstayid)])

### Table 28: respiratoryCharting
    respiratorychartzeros = np.zeros(len(eICU[27].patientunitstayid))
    for l27 in range(len(patient[k])):
        respiratorychartzeros = respiratorychartzeros + ((eICU[27].patientunitstayid == patient[k].patientunitstayid[l27])*1)

    respiratorychart.append(eICU[27][eICU[27].patientunitstayid == np.multiply(respiratorychartzeros, eICU[27].patientunitstayid)])

### Table 29: treatment
    treatmentzeros = np.zeros(len(eICU[28].patientunitstayid))
    for l28 in range(len(patient[k])):
        treatmentzeros = treatmentzeros + ((eICU[28].patientunitstayid == patient[k].patientunitstayid[l28])*1)

    treatment.append(eICU[28][eICU[28].patientunitstayid == np.multiply(treatmentzeros, eICU[28].patientunitstayid)])

### Table 30: vitalAperiodic
    vitalAzeros = np.zeros(len(eICU[29].patientunitstayid))
    for l29 in range(len(patient[k])):
        vitalAzeros = vitalAzeros + ((eICU[29].patientunitstayid == patient[k].patientunitstayid[l29])*1)

    vitalA.append(eICU[29][eICU[29].patientunitstayid == np.multiply(vitalAzeros, eICU[29].patientunitstayid)])

### Table 31: vitalPeriodic
    vitalPzeros = np.zeros(len(eICU[30].patientunitstayid))
    for l30 in range(len(patient[k])):
        vitalPzeros = vitalPzeros + ((eICU[30].patientunitstayid == patient[k].patientunitstayid[l30])*1)

    vitalP.append(eICU[30][eICU[30].patientunitstayid == np.multiply(vitalPzeros, eICU[30].patientunitstayid)])

############################################################################################################################
### Modification of each table based on the descriptions given on https://github.com/MIT-LCP/eicu-code/tree/master/notebooks
############################################################################################################################

### Table 1: admissiondrug
#eICU[0][(eICU[0].drughiclseqno == 550)]
druglist = ('metFormin', 'GliPizide', 'GlucoPHaGe', 'GLUcagon', 'inSULIN')
searchlist = []
diabdrugpat = []
for x1 in range(len(druglist)):
    searchlist.append([druglist[x1].lower(), druglist[x1].upper(), druglist[x1].capitalize()])
    diabdrugpat.append(eICU[0][(eICU[0].drugname.str.count(searchlist[x1][0]) | eICU[0].drugname.str.count(searchlist[x1][1]) | 
            eICU[0].drugname.str.count(searchlist[x1][2]) != 0)].dropna())

diabdrugpat = pd.concat(diabdrugpat[:], axis=0, join='outer', ignore_index=False)
diabdrugpat = diabdrugpat.sort_index()

### Table 2: admissionDx
eICU[1][(eICU[1].admitdxpath.str.count('docrine') == 1)]

### Table 3: allergy


### Table 4-6: apacheApsVar, apachePatientResult, apachePredVar
eICU[3][(eICU[3].temperature != -1)]

modmed = eICU[3][(eICU[3].eyes != -1)]
meds = (modmed.eyes + modmed.motor + modmed.verbal)/15
modmed = modmed.drop(modmed.columns[[5, 6, 7, 8]], axis=1)
modmed.insert(5, 'meds', meds ,True)


### Table 13: diagnosis
eICU[12][(eICU[12].diagnosisstring.str.count('diabet') == 1)]

### Table 15: infusionDrug
eICU[14][(eICU[14].drugname.str.count('insulin') == 1)]

### Table 16: intakeOutput
eICU[15][(eICU[15].celllabel.str.count('insulin') == 1)]

### Table 17: lab
eICU[16][(eICU[16].labname.str.count('bedside glucose') == 1)]

### Table 18: medication
eICU[17][(eICU[17].drughiclseqno == 807)]
eICU[17][(eICU[17].drughiclseqno == 19078)]
eICU[17][(eICU[17].drugname.str.count('GLUCA') == 1)]

### Table 23: nurseCharting
nursechart[12][(nursechart[12].nursingchartcelltypevallabel.str.count('Bedside Glucose') == 1)].sort_values(by = ['nursingchartoffset'])

### Table 24: pastHistory
pasthistory[142][(pasthistory[142].pasthistoryvalue.str.count('insulin') == 1)].sort_values(by = ['pasthistoryoffset'])

### Table 29: treatment
treatment[142][(treatment[142].treatmentstring.str.count('glucose') == 1)].sort_values(by = ['treatmentoffset'])

#import matplotlib.pyplot as plt
#a = vitalP[1].sort_values(by = ['observationoffset'])
#a[['heartrate', 'respiration']].plot()
#plt.legend(loc = 'upper right')

# mean_sao2 = sum(df[df.patientunitstayid == 141765].sao2.dropna())//len(df[df.patientunitstayid == 141765].sao2.dropna())
# isna().sum() = number of NaNs

#from scipy import interpolate
#A.respiration.interpolate(method = 'linear', limit_direction = 'both')