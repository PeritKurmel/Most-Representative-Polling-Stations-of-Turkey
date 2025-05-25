import pandas as pd
import numpy as np
import gurobipy
from gurobipy import GRB,Model,quicksum

listoffiles = {"2011":pd.read_excel('2011.xlsx'),"2015_1":pd.read_excel('2015_1.xlsx'),
"2015_2":pd.read_excel('2015_2.xlsx'),"2018":pd.read_excel('2018.xlsx'), "2023":pd.read_excel('2023.xlsx')}

run = 0
model_result= pd.DataFrame()

def convert_turkish_to_english(text):  #  Converts turkish characters to english
   turkish_to_english = {"ş":"s", "Ş":"S", "ı":"i", "İ":"I", "ğ":"g", "Ğ":"G", "ç":"c", "Ç":"C", "ö":"o", "Ö":"O", "ü":"u", "Ü":"U" }
   return text.translate(str.maketrans(turkish_to_english))


def function(data, target, station, il, ilçe):  # Prepare data for Polling Station

  if target == "Ülke":
    if station == "İl":
      data['Combined'] = data['İl']
    elif station == "İlçe":
      data['Combined'] = data['İl'] + "-" + data['İlçe']
    elif station == "Mahalle":
      data['Combined'] = data['İl'] + "-" + data['İlçe'] + "-" +  data['Mahalle']

  elif target == "İl" :
    data = data[data["İl"].str.contains(il)]
    if station == "İlçe":
      data['Combined'] = data['İl'] + "-" + data['İlçe']
      data = data[data['Combined'].str.contains(il)]
    elif station == "Mahalle":
      data['Combined'] = data['İl'] + "-" + data['İlçe'] + "-" +  data['Mahalle']
      data = data[data['Combined'].str.contains(il)]

  elif target == "İlçe" :
    data = data[data['İl'].str.contains(il)]
    data = data[data['İlçe'].str.contains(ilçe)]
    data['Combined'] = data['İl'] + "-" + data['İlçe'] + "-" +  data['Mahalle']
    data = data[data['Combined'].str.contains(il+ "-" +ilçe)]

  dataCopy = data.groupby(["Combined"]).sum()
  dataCopy = dataCopy.reset_index()
  dataCopy['Combined'] = dataCopy['Combined'].apply(convert_turkish_to_english)
  dataCopy = dataCopy.drop(columns=['İl','İlçe','Mahalle'])
  
  return dataCopy


def target_function(df, target, il, ilçe, P):
  B = ["İl","İlçe","Mahalle"]
  a = df['Combined'].str.split('-', expand=True)
  df_dropped = a.dropna(axis=1)
  dataframe = pd.DataFrame(df_dropped)
  for j in dataframe.columns.tolist():
    dataframe = dataframe.rename(columns={j: B[j]})
  data = pd.concat([df, dataframe], axis = 1)
  B.clear()

  if target == "Ülke":
    for j in range(len(P)): B.append(data[P[j]].sum())
  elif target == "İl" :
    data1 = data[data["İl"].str.contains(convert_turkish_to_english(il))]
    for j in range(len(P)): B.append(data1[P[j]].sum())
  elif target == "İlçe" :
    data2 = data[data['İl'].str.contains(convert_turkish_to_english(il))]
    data1 = data2[data2['İlçe'].str.contains(convert_turkish_to_english(ilçe))]
    for j in range(len(P)): B.append(data1[P[j]].sum())
  return B


def functionPercentage(df, party):
  df2 = df.copy()
  df = df.drop(columns=['Combined'])
  totalList = list(df.sum(axis=1))
  for j in range(len(party)):
    for i in df.index:
      df.loc[i,party[j]] = df.loc[i,party[j]] / totalList[i]
  df['Combined'] = df2["Combined"]
  df['Combined'] = df['Combined'].apply(convert_turkish_to_english)
  return df


def dynamically(B, P, data, year, voteParty):

  perc=[]

  for j in range(len(P)):
    perc.append(B[j]/sum(B))
    Station = gurobipy.multidict({(data['Combined'][i]): (data[P[j]][i])  for i in data.index  })
    voteParty.update({year + "-" + P[j] : Station[1]})
  return Station[0], perc

  
  
def Output(m):
    status_code = {1:'LOADED', 2:'OPTIMAL', 3:'INFEASIBLE', 4:'INF_OR_UNBD', 5:'UNBOUNDED'}
    status = m.status

    if status == 2:
        print('Optimal solution:')
        for v in m.getVars():
          if (v.x) >= 0.0000000000001:
            print(str(v.varName) + " = " + str(v.x))

            if (((str(v.varName)).split("["))[0] == "w" ):
              listofStations.append((((str(v.varName)).split("["))[1]).split("]")[0])
              listofWeights.append(v.x)
        print('Optimal objective value: ' + str(m.objVal) + "\n")
        objective_value.append(str(m.objVal))


def final(listofperc, years, votes, k):
    
    model=Model('final')
    model.setParam('OutputFlag',True)
    model.setParam('MIPGapAbs', 0.07)
    model.setParam('Timelimit', 20000)
    y = model.addVars(Station,  vtype=GRB.BINARY, name="y")  # if polling station selected or not
    w = model.addVars(Station, lb = 0, vtype=GRB.CONTINUOUS, name="w") # weights of polling stations
    Z = model.addVars(len(years), 10, vtype=GRB.CONTINUOUS, name="D") # difference of actual and selected results

    model.setObjective( quicksum(Z[i,j] for i in range(len(years)) for j in range(10) ), GRB.MINIMIZE) # Objective Function

    model.addConstr(quicksum(y) == k ) # Station Constraint
    model.addConstrs(w[i] <= 0.7*y[i] for i in Station) # Weight Constraint for big M
    model.addConstr(quicksum(w) == 1 ) # Weight Constraint (must sum up to 1)

    for i, year in zip(range(len(years)),years):
      for j in range(10):
        model.addConstr((w.prod(votes[year + "-" + parties[year][j]]) - listofperc[year][j] <= Z[i,j]))
      for j in range(10):
        model.addConstr((-w.prod(votes[year + "-" + parties[year][j]]) + listofperc[year][j] <= Z[i,j]))

    model.optimize()
    model.write("Model.lp")
    Output(model)


####################   CODE  ##########################

while True:

  elections = str(input('Hangi Seçimler(tireyle ayrılmış) ?\n'))
  years = elections.split("-")

  parties, B, listofperc, election_dataframes, df, votes  = {}, {}, {}, {}, {}, {}

  target = input('Ülke, İl, İlçe ?\n')   # input for target predict
  il, ilçe = "", ""

  if ((target == "İl")):
    il = input("Hangi İl?\n")
  elif (target == "İlçe"):
    name = input("Hangi İlçe (İl-İlçe formatında)? \n")
    name = name.split("-")
    il = (name)[0]
    ilçe = (name)[1]

  station = input('İl, İlçe, Mahalle ?\n')    # input for polling station

  for j in range(len(years)):
    df.update({years[j]: function((listoffiles[years[j]]), target, station, il, ilçe)})
    parties.update({years[j]:df[years[j]].columns.tolist()[1:]})
    election_dataframes.update({years[j]:df[years[j]]})
    common_stations = set(df[years[j]]['Combined'])

  # Iterate through remaining elections based on user selection and find common polling stations
  if (range(len(years))!= 1):
    for election in years[:-1] :
      common_stations = common_stations.intersection(set(election_dataframes[election]['Combined']))

  for index, election in enumerate(years):
    election_dataframes[election] = election_dataframes[election][election_dataframes[election]['Combined'].isin(list(common_stations))]
    election_dataframes[election] = election_dataframes[election].reset_index()
    election_dataframes[election] = election_dataframes[election].drop(columns=['index'])

    B.update({years[index] : target_function(election_dataframes[election], target, il, ilçe, parties[election])})
    election_dataframes[election] = functionPercentage(election_dataframes[election], parties[election])

  for election in years:
    Station, thePerc  = dynamically(B[election], parties[election], election_dataframes[election], election, votes )
    listofperc.update({election : thePerc})
  #print(listofperc["2018"]) 
  #x = election_dataframes[election].loc[election_dataframes[election]['Combined'] == "ANTALYA-KEPEZ-VARSAK KARSIYAKA MAH."]
  #print(election_dataframes[election].loc[election_dataframes[election]['Combined'] == "ANTALYA-KEPEZ-VARSAK KARSIYAKA MAH."])    
  for election in years:
    election_dataframes[election].to_excel(str(election)+'.xlsx')
  results={}
  k = input("Kaç yer seçilecek?\n")
  listofStations = []
  listofWeights = []
  objective_value = [None] * (int(k)-1)
  array = np.zeros((4, int(k)))
  final(listofperc, years, votes, int(k))

  model= pd.concat([pd.DataFrame({"Inputs": ["Elections: "+elections,"Input: "+ station,"Target: "+ target, il, ilçe]}),
          pd.DataFrame({ "Stations": listofStations, "Weights": listofWeights }), pd.DataFrame({"Objective Value": objective_value[::-1]} )], axis = 1)
  
  model_result = pd.concat([model_result, model], axis=0)

  word = input("Devam mı, Tamam mı?\n")
  if word == "Tamam":
        break
    
model_result.to_excel('output_tekseçim.xlsx')


'''
res = []  # sum of voteperc*weight of all selected stations for each party (a list of 10 elements) - [0.45, 0.24,...]
data = election_dataframes["2023"][election_dataframes["2023"]['Combined'].isin(listofStations)]
data = data.reset_index()

for j in range(len(parties["2023"])):
    res_party=[] # sum of voteperc*weight all selected stations for a specific party (a list of 5 elements)
    for i in data.index:
      res_party.append((data[parties["2023"][j]][i] * listofWeights[i]))
      res.append(sum(res_party))
    res_years.update({year: res})

Parties = []
Predicted = []
Actual = []
for i in range(len(res_years["year"])):
    Parties.append(str(parties[year][i]))
    Predicted.append(round((res_years[year][i]*sum(B[year]))))
    Actual.append(str(B[year][i]))
'''

