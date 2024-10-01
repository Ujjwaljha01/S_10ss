Practical 9:
Aim: Assosiation rule mining algorithm
from warnings import filterwarnings
filterwarnings("ignore")
    
import numpy as np
import  pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from apyori import apriori

df = pd.read_csv('D:\\STUDENT\\tycs82\\Groceries_dataset.csv')
print(df.head())
print(df.isnull().any())
all_products=df['itemDescription'].unique()
print("Total products: {}".format(len(all_products)))
def ditribution_plot(x,y,name=None,xaxis=None,yaxis=None):
    fig = go.Figure([go.Bar(x=x,y=y)])
    fig.update_layout(title_text=name,xaxis_title=xaxis,yaxis_title=yaxis)
    print(fig.show())

x = df['itemDescription'].value_counts()
x = x.sort_values(ascending = False)
x = x[:10]
ditribution_plot(x=x.index,y=x.values,yaxis="Count",xaxis="Products")

one_hot = pd.get_dummies(df['itemDescription'])
df.drop('itemDescription',inplace=True, axis=1)
df = df.join(one_hot)
print(df.head())
records=df.groupby(["Member_number","Date"])[all_products[:]].apply(sum)
records=records.reset_index()[all_products]

def get_Pnames(x):
    for product in all_products:
        if x[product]>0:
            x[product]=product
    return x
records=records.apply(get_Pnames, axis=1)
print(records.head())
x=records.values
x=[sub[~(sub==0)].tolist() for sub in x if sub[sub != 0].tolist()]
transactions=x

print(transactions[0:10])

rules = apriori(transactions,min_support=0.00030,min_confidance=0.05,min_lift=3,min_length=2,target="rules")
association_results=list(rules)

for item in association_results:
    pair=item[0]
    items=[x for x in pair]
    print("Rule: "+items[0] + " -> " + items[1])
    print("Support: "+str(item[1]))
    print("Confidence: "+str(item[2][0][2]))
    print("Lift: "+str(item[2][0][3]))
    print("========================================================\n\n")


Output:
===================== RESTART: D:/STUDENT/tycs82/pract9.py =====================
   Member_number        Date   itemDescription
0           1808  21-07-2015    tropical fruit
1           2552  05-01-2015        whole milk
2           2300  19-09-2015         pip fruit
3           1187  12-12-2015  other vegetables
4           3037  01-02-2015        whole milk
Member_number      False
Date               False
itemDescription    False
dtype: bool
Total products: 167
None
   Member_number        Date  ...  yogurt  zwieback
0           1808  21-07-2015  ...   False     False
1           2552  05-01-2015  ...   False     False
2           2300  19-09-2015  ...   False     False
3           1187  12-12-2015  ...   False     False
4           3037  01-02-2015  ...   False     False

[5 rows x 169 columns]
  tropical fruit  whole milk  ... toilet cleaner preservation products
0              0  whole milk  ...              0                     0
1              0  whole milk  ...              0                     0
2              0           0  ...              0                     0
3              0           0  ...              0                     0
4              0           0  ...              0                     0

[5 rows x 167 columns]
[['whole milk', 'yogurt', 'sausage', 'semi-finished bread'], ['whole milk', 'pastry', 'salty snack'], ['canned beer', 'misc. beverages'], ['sausage', 'hygiene articles'], ['soda', 'pickled vegetables'], ['frankfurter', 'curd'], ['whole milk', 'rolls/buns', 'sausage'], ['whole milk', 'soda'], ['beef', 'white bread'], ['frankfurter', 'soda', 'whipped/sour cream']]
Rule: frozen fish -> specialty chocolate
Support: 0.0003341575887188398
Confidence: 0.049019607843137254
Lift: 3.0689556157190907
========================================================


Rule: liver loaf -> fruit/vegetable juice
Support: 0.00040098910646260775
Confidence: 0.011787819253438114
Lift: 3.52762278978389
========================================================


Rule: ham -> pickled vegetables
Support: 0.0005346521419501437
Confidence: 0.03125
Lift: 3.4895055970149254
========================================================


Rule: meat -> roll products 
Support: 0.0003341575887188398
Confidence: 0.019841269841269844
Lift: 3.620547812620984
========================================================


Rule: salt -> misc. beverages
Support: 0.0003341575887188398
Confidence: 0.0211864406779661
Lift: 3.5619405827461437
========================================================


Rule: spread cheese -> misc. beverages
Support: 0.0003341575887188398
Confidence: 0.0211864406779661
Lift: 3.170127118644068
========================================================


Rule: seasonal products -> soups
Support: 0.0003341575887188398
Confidence: 0.04716981132075471
Lift: 14.704205974842766
========================================================


Rule: spread cheese -> sugar
Support: 0.00040098910646260775
Confidence: 0.06
Lift: 3.3878490566037733
========================================================


Rule: butter -> bottled beer
Support: 0.0003341575887188398
Confidence: 0.007374631268436578
Lift: 3.8050554368833285
========================================================


Rule: hard cheese -> whole milk
Support: 0.0003341575887188398
Confidence: 0.007374631268436578
Lift: 3.9409502739148756
========================================================


Rule: frozen vegetables -> canned beer
Support: 0.0003341575887188398
Confidence: 0.008880994671403198
Lift: 6.644316163410303
========================================================


Rule: canned beer -> sausage
Support: 0.00040098910646260775
Confidence: 0.010657193605683837
Lift: 4.309826700590467
========================================================


Rule: butter -> soda
Support: 0.0003341575887188398
Confidence: 0.009487666034155597
Lift: 3.086172758023265
========================================================


Rule: butter milk -> yogurt
Support: 0.0003341575887188398
Confidence: 0.019011406844106463
Lift: 4.9046151829028455
========================================================


Rule: frozen vegetables -> canned beer
Support: 0.0003341575887188398
Confidence: 0.007122507122507123
Lift: 3.437873357228196
========================================================


Rule: whole milk -> canned beer
Support: 0.00040098910646260775
Confidence: 0.008547008547008546
Lift: 4.918803418803418
========================================================


Rule: soda -> chewing gum
Support: 0.00040098910646260775
Confidence: 0.03333333333333333
Lift: 5.732950191570881
========================================================


Rule: pork -> yogurt
Support: 0.00040098910646260775
Confidence: 0.004669260700389105
Lift: 3.4933073929961087
========================================================


Rule: coffee -> frankfurter
Support: 0.0003341575887188398
Confidence: 0.008849557522123895
Lift: 3.6782202556538843
========================================================


Rule: coffee -> frankfurter
Support: 0.0003341575887188398
Confidence: 0.010570824524312896
Lift: 3.438505377332475
========================================================


Rule: coffee -> sausage
Support: 0.0003341575887188398
Confidence: 0.010570824524312896
Lift: 3.2952343199436225
========================================================


Rule: curd -> fruit/vegetable juice
Support: 0.0003341575887188398
Confidence: 0.009920634920634922
Lift: 5.497868900646679
========================================================


Rule: curd -> margarine
Support: 0.0003341575887188398
Confidence: 0.009920634920634922
Lift: 5.301516439909298
========================================================


Rule: curd -> yogurt
Support: 0.00046782062420637575
Confidence: 0.007751937984496124
Lift: 3.4115367077063383
========================================================


Rule: hard cheese -> sausage
Support: 0.0003341575887188398
Confidence: 0.022727272727272728
Lift: 3.7785353535353536
========================================================


Rule: pip fruit -> rolls/buns
Support: 0.0003341575887188398
Confidence: 0.022026431718061675
Lift: 4.453804024288606
========================================================


Rule: root vegetables -> margarine
Support: 0.0003341575887188398
Confidence: 0.01037344398340249
Lift: 3.1043568464730287
========================================================


Rule: yogurt -> margarine
Support: 0.00040098910646260775
Confidence: 0.0066445182724252485
Lift: 3.106935215946843
========================================================


Rule: newspapers -> sausage
Support: 0.0003341575887188398
Confidence: 0.006459948320413437
Lift: 3.580007656235047
========================================================


Rule: onions -> yogurt
Support: 0.0003341575887188398
Confidence: 0.016501650165016504
Lift: 3.1655665566556657
========================================================


Rule: waffles -> sausage
Support: 0.0003341575887188398
Confidence: 0.002736726874657909
Lift: 3.4124703521255246
========================================================


Rule: soft cheese -> yogurt
Support: 0.0003341575887188398
Confidence: 0.03333333333333333
Lift: 4.122038567493113
========================================================


Rule: pork -> yogurt
Support: 0.00040098910646260775
Confidence: 0.004669260700389105
Lift: 3.037658602605312
========================================================


Rule: soda -> pastry
Support: 0.0003341575887188398
Confidence: 0.006459948320413437
Lift: 5.685894512843897
========================================================


Rule: yogurt -> rolls/buns
Support: 0.0003341575887188398
Confidence: 0.005537098560354374
Lift: 4.142580287929125
========================================================
