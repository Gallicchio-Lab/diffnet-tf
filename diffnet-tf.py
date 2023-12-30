import numpy as np
import random, math, re, sys
import tensorflow as tf
import pandas as pd

#
# Extended DiffNet implementation with TensorFlow
#
# Usage:
#  python diffnet-tf.py < data.csv
#


#transid,DGb,DGberr,minsample,maxsample,optimize
dgdata = pd.read_csv(sys.stdin, header=0, delimiter=",", comment="#")
print(dgdata.to_string())

pattern = re.compile('([^:\|]*):([^:\|]*)\|([^:\|]*):([^:\|]*)')

g = {}
variables = []
for key,dgb,opt in zip(dgdata['transid'],dgdata['DGb'],dgdata['optimize']):
    matches = pattern.search(key)
    rcpt1 = matches.group(1)
    lig1  = matches.group(2)
    rcpt2 = matches.group(3)
    lig2  = matches.group(4)
    if rcpt2 == "" and lig2 == "": #ABFEs
        g[key] = tf.Variable(dgb, dtype=tf.float64)
        if opt == 'T':
            variables.append(g[key])

a = {}
s = {}
for key,dgb,dgberr,opt in zip(dgdata['transid'],dgdata['DGb'],dgdata['DGberr'],dgdata['optimize']):
    if opt != 'T':
        a[key] = tf.constant(float(dgb), dtype=tf.float64)
        s[key] = tf.constant(float(dgberr), dtype=tf.float64)

ar = { key:tf.constant( 0.0, dtype=tf.float64) for key in a }

@tf.function
def cost(g, a, ar):
    cw = []
    # pattern is "rcpt2:lig2|rcpt1:lig1"
    pattern = re.compile('([^:\|]*):([^:\|]*)\|([^:\|]*):([^:\|]*)')
    for key, value in a.items(): 
        matches = pattern.search(key)
        rcpt1 = matches.group(1)
        lig1  = matches.group(2)
        rcpt2 = matches.group(3)
        lig2  = matches.group(4)
        av = a[key] + ar[key]
        if rcpt2 != "" and lig2 != "":
            if rcpt1 != rcpt2:
                if lig1 != lig2:
                    #case D swapping
                    jb = rcpt2 + ":" + lig2 + "|:"
                    ib = rcpt2 + ":" + lig1 + "|:"
                    ja = rcpt1 + ":" + lig2 + "|:"
                    ia = rcpt1 + ":" + lig1 + "|:"
                    cw.append( ( -(g[jb]-g[ib])+(g[ja]-g[ia]) - av )/s[key] )
                else:
                    #case C hopping
                    ib = rcpt2 + ":" + lig1 + "|:"
                    ia = rcpt1 + ":" + lig1 + "|:"
                    cw.append( ( (g[ib]-g[ia]) - av )/s[key] )
            else:
                # case B RBFE
                ja = rcpt1 + ":" + lig2 + "|:"
                ia = rcpt1 + ":" + lig1 + "|:"
                cw.append( ( (g[ja]-g[ia]) - av )/s[key] )
        else:
            # case A ABFE
            ia = rcpt1 + ":" + lig1 + "|:"
            cw.append( ( g[ia] - av )/s[key] )
    c = tf.convert_to_tensor( cw )
    cost = tf.tensordot(c,c,axes=1)
    return cost

optimizer = tf.keras.optimizers.Adam(0.1)


epochs = 1000
for _ in range(epochs):
    with tf.GradientTape() as tp:
        #your loss/cost function must always be contained within the gradient tape instantiation
        costf = cost(g, a, ar)
    gradients = tp.gradient(costf, variables)
    optimizer.apply_gradients(zip(gradients, variables))

gbest = { key:g[key].numpy() for key in g }

g2 = { key:0.0 for key in g  }

#error analysis
nrounds = 20
for rounds in range(nrounds):
    with tf.GradientTape() as tp:
        ar = { key:tf.random.normal([1],stddev=s[key], dtype=tf.float64)[0] for key in a   }
    epochs = 1000
    for _ in range(epochs):
        with tf.GradientTape() as tp:
            #your loss/cost function must always be contained within the gradient tape instantiation
            costf = cost(g, a, ar)
        gradients = tp.gradient(costf, variables)
        optimizer.apply_gradients(zip(gradients, variables))
    
    gg = { key:g[key].numpy() for key in g }
    for key in g2:
        g2[key] += np.power(gg[key]-gbest[key],2)/float(nrounds)

gerr = { key:np.sqrt(g2[key]) for key in g2  }

for key in gbest:
    print(key, gbest[key], "+-", gerr[key])
